import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import shutil

# 数据路径
data_dir = r"./data_source"  # 替换为你的数据路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 数据预处理
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# 加载数据集
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 模型设置
num_classes = 4  # 假设有4个类别
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
model.to(DEVICE)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
lr_18 = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr_18, momentum=0.9)
error_all_images = []


# 训练函数
def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25
):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_id + 1) % 50 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        (batch_id + 1) * len(inputs),
                        len(train_loader.dataset),
                        100.0 * (batch_id + 1) / len(train_loader),
                        loss.item(),
                    )
                )
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")
        test_model(model, val_loader, DEVICE)


# 验证函数
def test_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    error_iter_images = []
    batch_size = 0
    with torch.no_grad():
        j = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
            test_loss += loss.item()

            # 找出预测错误的图像
            batch_size = inputs.size(0)
            for i in range(batch_size):
                if preds[i] != labels[i]:
                    error_iter_images.append(val_loader.dataset.imgs[i + 64 * j][0])
            j += 1

    acc = correct.double() / len(val_loader.dataset)
    avgloss = test_loss / len(val_loader)
    print(
        "\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            avgloss, correct, len(val_loader.dataset), 100 * acc
        )
    )


# 训练和验证模型
train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE)
test_model(model, val_loader, DEVICE)

# 保存模型
torch.save(model, f"base_model_18_{lr_18}_nolaji_source.pth")

# # 找到每个列表都包含的元素
# common_elements = set(error_all_images[0])
# for sublist in error_all_images[1:]:
#     common_elements.intersection_update(sublist)
#
# # 输出结果
# print(list(common_elements))
# error_all_images = list(common_elements)
# # 替换反斜杠为正斜杠
# error_all_images = [path.replace('\\', '/') for path in error_all_images]
#
# # 目标根目录
# target_root_dir = './source_lajidata'
#
# # 遍历每个图像路径并移动到目标目录
# for image_path in error_all_images:
#     # 获取类别名和文件名
#     category = os.path.basename(os.path.dirname(image_path))
#     file_name = os.path.basename(image_path)
#
#     # 创建目标类别目录
#     target_dir = os.path.join(target_root_dir, category)
#     os.makedirs(target_dir, exist_ok=True)
#
#     # 移动文件
#     target_path = os.path.join(target_dir, file_name)
#     shutil.move(image_path, target_path)
#
# print("所有图像已成功移动。")
