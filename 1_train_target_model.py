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

#训练受害者模型

# 设置随机种子，保证每次运行代码时，生成的随机数都是一样的。
# 这对于复现实验结果至关重要。
torch.manual_seed(42)
np.random.seed(42)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")



# --- 数据加载与预处理 ---

# 定义训练集的图像预处理流程。这是一个标准的“代码模板”。
# transforms.Compose 会将一系列处理步骤串联起来。
#在训练集中会使用数据增强技术
transform_train = transforms.Compose([
    # 1. 随机裁剪：将32x32的图像随机裁剪，四周填充4个像素。这能增加图像的多样性，防止模型过拟合。
    transforms.RandomCrop(32, padding=4),
    # 2. 随机水平翻转：以50%的概率将图像水平翻转。
    transforms.RandomHorizontalFlip(),
    # 3. 转换为Tensor：将PIL图像或Numpy数组转换为PyTorch的Tensor格式，并将像素值从[0, 255]缩放到[0, 1]。
    transforms.ToTensor(),
    # 4. 归一化：使用CIFAR-10数据集的均值和标准差对图像进行归一化。
    # 这可以加速模型收敛。这三个数分别是R, G, B三个通道的均值和标准差。
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# 定义测试集的图像预处理流程。
# 注意：测试集通常不做数据增强（如随机裁剪、翻转），只做必要的格式转换和归一化，以保证测试结果的确定性。
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# 加载CIFAR-10训练数据集。
# root='./data'：数据集存放的路径。
# train=True：表示加载训练集。
# download=True：如果路径下没有数据集，就自动下载。
# transform=transform_train：应用我们上面定义的训练集预处理。
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# 创建数据加载器（DataLoader）。它能帮我们自动打包数据成一批一批(batch)，并在每个epoch开始时打乱数据。
# batch_size=128：每次从数据集中取128张图片进行训练。
# shuffle=True：打乱数据顺序。
# num_workers=0：使用主进程加载数据。如果设为大于0的数，会使用多线程，加快数据加载速度（在Windows上可能需要特殊设置）。
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

# 加载CIFAR-10测试数据集，与训练集类似。
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)


# --- 模型定义 ---
# 定义目标模型（ResNet-18），定义一个函数来创建我们的目标模型。
def create_victim_model():

    # 加载一个ResNet-18模型。pretrained=False 表示我们不要别人预训练好的权重，我们要从零开始训练。
    model = models.resnet18(pretrained=False)
    # --- 对ResNet-18进行微调以适应CIFAR-10 ---
    # ResNet原文是为ImageNet（224x224图像）设计的，CIFAR-10图像只有32x32，太小了。
    # 直接用原版ResNet会导致特征图变得过小，信息丢失严重。所以需要做一些修改。

    # 1. 修改第一个卷积层：原始的kernel_size=7, stride=2，对于32x32的图来说太大了。
    # 改为kernel_size=3, stride=1，更适合小图像。
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 2. 移除最大池化层：nn.Identity() 是一个“占位符”，它什么都不做，直接返回输入。
    # 这样可以避免特征图尺寸过早地减半。
    model.maxpool = nn.Identity()  # 移除maxpool层，因为CIFAR-10图像较小
    # 3. 修改最后的全连接层（分类头）：
    # model.fc.in_features 获取到全连接层输入特征的数量。
    # nn.Linear(...) 创建一个新的全连接层，输出维度为10，因为CIFAR-10有10个类别。
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10个类别
    return model


# --- 训练与评估 ---
# 定义训练模型的函数。这是一个非常标准的PyTorch训练代码模板。
def train_victim_model(model, epochs=100, save_path='victim_model.pth'):#函数内部的循环执行100次
    """训练目标模型并保存"""
    model = model.to(device) # 将模型移动到我们之前定义的设备（GPU或CPU）上
    criterion = nn.CrossEntropyLoss()# 定义损失函数。交叉熵损失是分类任务最常用的损失函数。
    # 定义优化器。SGD（随机梯度下降）是经典的优化器。
    # lr=0.1：学习率，控制每次参数更新的步长。
    # momentum=0.9：动量，帮助加速梯度下降。
    # weight_decay=5e-4：权重衰减，一种正则化手段，防止过拟合。
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # 定义学习率调度器。CosineAnnealingLR会让学习率按照余弦曲线进行变化，有助于模型跳出局部最优。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0 # 用于记录最佳测试准确率
    print("开始训练目标模型...")

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(epochs):
        model.train() # 告诉模型现在是训练模式。这会启用Dropout和BatchNorm等层。
        running_loss = 0.0
        correct = 0
        total = 0

        # 内层循环，遍历训练数据加载器中的每一个batch
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device) # 将数据也移动到GPU/CPU

            optimizer.zero_grad() # 清空上一轮的梯度
            outputs = model(inputs) # 前向传播：将输入数据喂给模型，得到输出
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播：计算梯度
            optimizer.step() # 更新模型参数

            # --- 统计损失和准确率，用于打印日志 ---
            running_loss += loss.item()#把当前这一批数据的损失值，累加到一个“进行中的总损失”里
            _, predicted = outputs.max(1) # 获取预测结果中概率最大的那个类别的索引，最大值是多少并不关心
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() # 统计预测正确的数量

            if (i + 1) % 100 == 0: # 每训练100个batch打印一次信息
                print(
                    f'Epoch: {epoch + 1}/{epochs}, Batch: {i + 1}/{len(trainloader)}, Loss: {running_loss / 100:.4f}, Acc: {100. * correct / total:.2f}%')
                running_loss = 0.0 # 重置running_loss

        # 在每个epoch结束后评估模型
        acc = test_model(model, testloader)
        scheduler.step()  # 更新学习率

        # 保存最佳模型 ,如果当前准确率比历史最佳准确率要高，就保存模型
        if acc > best_acc:
            best_acc = acc
            # torch.save 保存模型的状态字典（参数）。只保存参数比保存整个模型更灵活、更推荐。
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存，准确率: {acc:.2f}%')

    print(f"训练完成！最佳准确率: {best_acc:.2f}%")
    return model

# 定义测试模型的函数。这也是一个标准模板。
def test_model(model, dataloader):
    """测试模型准确率"""
    model.eval() # 告诉模型现在是评估模式。这会关闭Dropout和BatchNorm等层。
    correct = 0
    total = 0

    #这里没有计算loss，没有loss.backward()，也没有optimizer.step()。因为监考老师只负责打分，不负责在考场上给学生讲题和辅导。
    with torch.no_grad(): # 在这个代码块中，所有计算都不会记录梯度，可以节省显存，加速计算。
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f'测试准确率: {acc:.2f}%')
    return acc


# --- 主程序入口 ---
# 只有当你直接运行这个.py文件时，下面的代码才会被执行。
# 如果你把它当作一个模块导入到其他文件中，下面的代码不会运行。
if __name__ == "__main__":
    VICTIM_MODEL_DIR = "results/0_victim_model"
    VICTIM_MODEL_PATH = os.path.join(VICTIM_MODEL_DIR, "victim_model.pth")

    victim_model = create_victim_model()# 创建模型实例
    # 开始训练，epochs=50表示我们只训练50个周期来快速得到一个模型，请把整个训练过程重复50次，模型会把整个CIFAR-10训练集（50,000张图片）完整地看 50遍。每看完一遍，就完成一个Epoch。
    trained_model = train_victim_model(victim_model, epochs=50, save_path=VICTIM_MODEL_PATH)

    # 训练结束后，加载保存的那个最佳模型
    best_model = create_victim_model()
    best_model.load_state_dict(torch.load(VICTIM_MODEL_PATH))
    best_model = best_model.to(device)
    # 在测试集上再次验证最佳模型的性能
    test_model(best_model, testloader)
