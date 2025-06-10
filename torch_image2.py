import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

# 数据路径
data_dir = r"./data_source2"  # 替换为你的数据路径
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
#
# # # 加载数据集
# # train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
# # val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=transform)
# #
# # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# #
# # # 模型设置
# # num_classes = 4  # 假设有4个类别
# # model = models.resnet18(pretrained=False)
# # model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
# # model.to(DEVICE)
# #
# # # 损失函数和优化器
# # criterion = nn.CrossEntropyLoss()
# # lr_18 = 0.1
# # optimizer = torch.optim.SGD(model.parameters(), lr=lr_18, momentum=0.9)
# #
# #
# # # 训练函数
# # def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=40,
# #                 early_stopping_patience=10):
# #     best_val_loss = float('inf')
# #     patience_counter = 0
# #
# #     for epoch in range(num_epochs):
# #         model.train()
# #         running_loss = 0.0
# #         for batch_id, (inputs, labels) in enumerate(train_loader):
# #             inputs, labels = inputs.to(device), labels.to(device)
# #             optimizer.zero_grad()
# #             outputs = model(inputs)
# #             loss = criterion(outputs, labels)
# #             loss.backward()
# #             optimizer.step()
# #             running_loss += loss.item()
# #             if (batch_id + 1) % 50 == 0:
# #                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
# #                     epoch, (batch_id + 1) * len(inputs), len(train_loader.dataset),
# #                            100. * (batch_id + 1) / len(train_loader), loss.item()))
# #
# #         epoch_loss = running_loss / len(train_loader)
# #         print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}')
# #
# #         val_loss, val_acc = test_model(model, val_loader, DEVICE)
# #         print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
# #
# #         # Early stopping
# #         if val_loss < best_val_loss:
# #             best_val_loss = val_loss
# #             patience_counter = 0
# #             torch.save(model.state_dict(), 'best_model_source.pth')  # 保存最好的模型
# #         else:
# #             patience_counter += 1
# #             if patience_counter >= early_stopping_patience:
# #                 print("Early stopping triggered")
# #                 break
# #
# #         # 学习率调整
# #         for param_group in optimizer.param_groups:
# #             param_group['lr'] = param_group['lr'] * 0.9  # 每个epoch后将学习率减小10%
# #
# #     # 加载最好的模型
# #     model.load_state_dict(torch.load('best_model.pth'))
# # 加载整个模型
model = torch.load(
    r"D:\source\python\torch_big_data\data_vision\fuquqi_base_model_18_0.1_nolaji_source.pth"
)
model.to(DEVICE)
#
# # # 验证函数
# # def test_model(model, val_loader, device):
# #     model.eval()
# #     correct = 0
# #     total = 0
# #     test_loss = 0
# #     with torch.no_grad():
# #         for inputs, labels in val_loader:
# #             inputs, labels = inputs.to(device), labels.to(device)
# #             outputs = model(inputs)
# #             loss = criterion(outputs, labels)
# #             _, preds = torch.max(outputs, 1)
# #             correct += torch.sum(preds == labels)
# #             test_loss += loss.item()
# #     acc = correct.double() / len(val_loader.dataset)
# #     avgloss = test_loss / len(val_loader)
# #     return avgloss, acc
# #
# #
# # # 训练和验证模型
# # train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE)
#
# # 如果需要预测test数据集，可以加载test数据集并进行预测
# test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
#
# # 预测函数
# # 预测函数
# def predict(model, test_loader, device):
#     model.eval()
#     predictions = []
#     true_labels = []
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             predictions.extend(preds.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())
#     return true_labels, predictions
#
# # 进行预测
# true_labels, predictions = predict(model, test_loader, DEVICE)
#
# # 转换为 numpy 数组
# true_labels = np.array(true_labels)
# predictions = np.array(predictions)
# # 计算准确率
# accuracy = np.mean(true_labels == predictions)
# # 定义标签映射字典
# label_map = {
#     0: "Cyclone",
#     1: "Earthquake",
#     2: "Flood",
#     3: "Wildfire"
# }
# # 打开文件并写入数据
# with open('image_batch_results.txt', 'w', encoding='utf-8') as file:
#     # 写入列标题
#     file.write("真实值\t预测值\n")
#     # 写入数据，替换标签值为真实名称
#     for true, pred in zip(true_labels, predictions):
#         true_name = label_map.get(true, "未知")
#         pred_name = label_map.get(pred, "未知")
#         file.write(f"{true_name}\t{pred_name}\n")
#
# print(f"结果已写入 results.txt,{accuracy}")


import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# 定义标签映射字典
label_map = {0: "Cyclone", 1: "Earthquake", 2: "Flood", 3: "Wildfire"}


# 扩展 ImageFolder 类以获取文件名
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.imgs[index][0]  # 获取图像的路径
        return image, label, path


# 预测函数
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    image_paths = []
    incorrect_ids = []

    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            image_paths.extend(paths)

            # 记录错误的图像路径
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    incorrect_ids.append(paths[i])

    return true_labels, predictions, incorrect_ids


#
# # 数据预处理和数据加载
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])

# data_dir = './深度学习模型/test'  # 替换为你的数据路径
test_dataset = CustomImageFolder(root=f"{data_dir}/val", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 进行预测
true_labels, predictions, incorrect_ids = predict(model, test_loader, DEVICE)

# 转换为 numpy 数组
true_labels = np.array(true_labels)
predictions = np.array(predictions)

# 计算准确率
accuracy = np.mean(true_labels == predictions)
print(accuracy)
# # 目标文件夹路径
# target_folder = './垃圾图像数据'
# import shutil
# import os
# # 确保目标文件夹存在
# os.makedirs(target_folder, exist_ok=True)
# for path in incorrect_ids:
#     # 提取文件名
#     file_name = os.path.basename(path)
#
#     # 构建目标文件路径
#     target_path = os.path.join(target_folder, file_name)
#
#     # 移动文件
#     shutil.move(path, target_path)
#     print(f"已移动文件 {path} 到 {target_path}")
#     print(f"{path}\n")

print("预测错误的图像路径已写入 ./深度学习模型/incorrect_images.txt")
