import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from pathlib import Path
import os
import shutil
from typing import List, Tuple, Optional

def setup_device() -> torch.device:
    """设置计算设备（GPU或CPU）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

def create_data_loaders(data_dir: str, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        data_dir (str): 数据集根目录路径
        batch_size (int): 批次大小，默认为64

    Returns:
        Tuple[DataLoader, DataLoader]: 训练和验证数据加载器
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
        val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    except Exception as e:
        raise ValueError(f"加载数据集失败: {e}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def initialize_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    初始化 ResNet18 模型

    Args:
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练权重，默认为 False

    Returns:
        nn.Module: 初始化好的模型
    """
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
    return model

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 25
) -> None:
    """
    训练模型并在每个 epoch 后进行验证

    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数，默认为25
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_id, (inputs, labels) in enumerate(train_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (batch_id + 1) % 50 == 0:
                    print(
                        f"训练 Epoch: {epoch} [{(batch_id + 1) * len(inputs)}/{len(train_loader.dataset)} "
                        f"({100. * (batch_id + 1) / total_batches:.0f}%)]\t损失: {loss.item():.6f}"
                    )
            except Exception as e:
                print(f"训练批次 {batch_id} 出错: {e}")
                continue

        epoch_loss = running_loss / total_batches
        print(f"Epoch {epoch}/{num_epochs - 1}, 平均损失: {epoch_loss:.4f}")
        test_model(model, val_loader, device)

def test_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> None:
    """
    验证模型性能并记录预测错误的图像

    Args:
        model: 神经网络模型
        val_loader: 验证数据加载器
        device: 计算设备
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    error_images = []

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(val_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

                # 记录预测错误的图像路径
                for i in range(len(preds)):
                    if preds[i] != labels[i]:
                        image_idx = i + batch_id * val_loader.batch_size
                        if image_idx < len(val_loader.dataset.imgs):
                            error_images.append(val_loader.dataset.imgs[image_idx][0])
            except Exception as e:
                print(f"验证批次 {batch_id} 出错: {e}")
                continue

    accuracy = correct / total if total > 0 else 0
    avg_loss = test_loss / len(val_loader) if len(val_loader) > 0 else 0
    print(
        f"\n验证集: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({100 * accuracy:.0f}%)\n"
    )
    return error_images

def move_error_images(error_images: List[str], target_root_dir: str) -> None:
    """
    移动预测错误的图像到目标目录

    Args:
        error_images: 预测错误的图像路径列表
        target_root_dir: 目标根目录
    """
    target_root = Path(target_root_dir)
    for image_path in error_images:
        try:
            image_path = Path(image_path.replace('\\', '/'))
            category = image_path.parent.name
            file_name = image_path.name
            target_dir = target_root / category
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(target_dir / file_name))
        except Exception as e:
            print(f"移动图像 {image_path} 失败: {e}")
    print("所有错误预测图像已移动")

def main():
    """主函数，执行模型训练、验证和错误图像处理"""
    try:
        # 配置参数
        data_dir = "./data" #bug
        target_root_dir = "./trash"#bug
        num_classes = 4
        learning_rate = 0.1
        num_epochs = 25
        batch_size = 64

        # 初始化
        device = setup_device()
        train_loader, val_loader = create_data_loaders(data_dir, batch_size)
        model = initialize_model(num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # 训练和验证
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
        error_images = test_model(model, val_loader, device)

        # 保存模型
        model_path = f"./base_model_18_{learning_rate}_source.pth" #bug
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")

        # 移动错误预测的图像
        if error_images:
            move_error_images(error_images, target_root_dir)
        else:
            print("没有发现预测错误的图像")

    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()