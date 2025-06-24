import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)


# 定义目标模型（ResNet-18）
def create_victim_model():
    # 修改ResNet-18以适应CIFAR-10
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除maxpool层，因为CIFAR-10图像较小
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10个类别
    return model


def train_victim_model(model, epochs=100, save_path='victim_model.pth'):
    """训练目标模型并保存"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    print("开始训练目标模型...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch: {epoch + 1}/{epochs}, Batch: {i + 1}/{len(trainloader)}, Loss: {running_loss / 100:.4f}, Acc: {100. * correct / total:.2f}%')
                running_loss = 0.0

        # 在每个epoch结束后评估模型
        acc = test_model(model, testloader)
        scheduler.step()

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存，准确率: {acc:.2f}%')

    print(f"训练完成！最佳准确率: {best_acc:.2f}%")
    return model


def test_model(model, dataloader):
    """测试模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f'测试准确率: {acc:.2f}%')
    return acc


# 训练和保存目标模型
if __name__ == "__main__":
    victim_model = create_victim_model()
    trained_model = train_victim_model(victim_model, epochs=50)

    # 加载并测试最佳模型
    best_model = create_victim_model()
    best_model.load_state_dict(torch.load('victim_model.pth'))
    best_model = best_model.to(device)
    test_model(best_model, testloader)