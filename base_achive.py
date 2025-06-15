import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

def initialize_device():
    """
    初始化计算设备（GPU 或 CPU）

    返回:
        torch.device: 可用的计算设备
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

def create_data_loaders(data_dir, batch_size=64):
    """
    创建训练和验证数据加载器

    参数:
        data_dir (str): 数据集根目录路径
        batch_size (int): 批次大小，默认为 64

    返回:
        tuple: (训练数据加载器, 验证数据加载器, 类别数)
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
        val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        num_classes = len(train_dataset.classes)
        print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 类别数: {num_classes}")
        
        return train_loader, val_loader, num_classes
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise

def initialize_model(model_name, num_classes, pretrained=False):
    """
    初始化模型并将其移到指定设备

    参数:
        model_name (str): 模型名称（如 'resnet18', 'resnet34'）
        num_classes (int): 分类类别数
        pretrained (bool): 是否使用预训练权重，默认为 False

    返回:
        nn.Module: 初始化后的模型
    """
    try:
        model = getattr(models, model_name)(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        return model.to(DEVICE)
    except AttributeError:
        print(f"模型 {model_name} 不存在")
        raise
    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    训练模型

    参数:
        model (nn.Module): 待训练的模型
        train_loader (DataLoader): 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device (torch.device): 计算设备
        num_epochs (int): 训练轮数，默认为 10
    """
    total_num = len(train_loader.dataset)
    print(f"训练集总数: {total_num}, 批次数量: {len(train_loader)}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_id, (inputs, labels) in enumerate(train_loader):
            try:
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (batch_id + 1) % 50 == 0:
                    print(
                        f"训练 Epoch: {epoch} [{(batch_id + 1) * len(inputs)}/{total_num} "
                        f"({100.0 * (batch_id + 1) / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}"
                    )
            except Exception as e:
                print(f"批次 {batch_id} 训练失败: {e}")
                continue
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs - 1}, 平均损失: {epoch_loss:.4f}")
        test_model(model, val_loader, device)

def test_model(model, val_loader, device):
    """
    测试模型

    参数:
        model (nn.Module): 待测试的模型
        val_loader (DataLoader): 验证数据加载器
        device (torch.device): 计算设备
    """
    model.eval()
    correct = 0
    total_num = len(val_loader.dataset)
    test_loss = 0.0
    
    print(f"验证集总数: {total_num}, 批次数量: {len(val_loader)}")
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            try:
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
            except Exception as e:
                print(f"验证批次处理失败: {e}")
                continue
    
    accuracy = correct / total_num
    avg_loss = test_loss / len(val_loader)
    print(f"\n验证集: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total_num} ({100 * accuracy:.0f}%)\n")

def save_model(model, model_name, lr, save_dir="models"):
    """
    保存模型

    参数:
        model (nn.Module): 待保存的模型
        model_name (str): 模型名称
        lr (float): 学习率
        save_dir (str): 保存目录，默认为 "models"
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"base_model_{model_name}_{lr}.pth")
        torch.save(model, save_path)
        print(f"模型保存至: {save_path}")
    except Exception as e:
        print(f"模型保存失败: {e}")

def main():
    """
    主函数：初始化数据、模型、训练并保存
    """
    global DEVICE, train_loader, val_loader, criterion
    # 初始化设备和数据
    DEVICE = initialize_device()
    data_dir = r"D:\source\python\MNIST_torch\deep_learn_class_devise\afhq"
    train_loader, val_loader, num_classes = create_data_loaders(data_dir)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义模型配置
    models_config = [
        {"name": "resnet18", "lr": 0.1},
        {"name": "resnet34", "lr": 0.01}
    ]

    for config in models_config:
        try:
            # 初始化模型和优化器
            model = initialize_model(config["name"], num_classes, pretrained=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
            
            # 训练和测试
            print(f"\n开始训练 {config['name']}...")
            train_model(model, train_loader, criterion, optimizer, DEVICE)
            print(f"测试 {config['name']}...")
            test_model(model, val_loader, DEVICE)
            
            # 保存模型
            save_model(model, config["name"], config["lr"])
        except Exception as e:
            print(f"处理模型 {config['name']} 失败: {e}")
            continue

if __name__ == "__main__":
    main()