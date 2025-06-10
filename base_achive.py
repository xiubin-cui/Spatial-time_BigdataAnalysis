import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable

data_dir = (
    r"D:\source\python\MNIST_torch\deep_learn_class_devise\afhq"  # 替换为你的数据路径
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
print(DEVICE)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
num_classes = 4  # 假设有3个类别
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
lr_18 = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr_18, momentum=0.9)


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    print(device)
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # running_loss += loss.item() * inputs.size(0)
            print_loss = loss.data.item()
            running_loss += print_loss
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


def test_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    total_num = len(val_loader.dataset)
    print(total_num, len(val_loader))
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
            print_loss = loss.data.item()
            test_loss += print_loss
    correct = correct.data.item()
    acc = correct / total_num
    avgloss = test_loss / len(val_loader)
    print(
        "\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            avgloss, correct, len(val_loader.dataset), 100 * acc
        )
    )


print(DEVICE)
train_model(model, train_loader, criterion, optimizer, DEVICE)
test_model(model, val_loader, DEVICE)
torch.save(model, f"base_model_18{lr_18}.pth")


#
# # 增加两个模型
# model1 = models.resnet34(pretrained=False)
#
# model1.fc = nn.Linear(model1.fc.in_features, num_classes, bias=True)
# model1.to(DEVICE)
# lr_34 = 0.01
#
# optimizer1 = torch.optim.SGD(model1.parameters(), lr=lr_34, momentum=0.9)
#
# train_model(model1, train_loader, criterion, optimizer1, DEVICE)
#
# test_model(model1, val_loader, DEVICE)

# torch.save(model1, f'base_model_34_{lr_34}.pth')
