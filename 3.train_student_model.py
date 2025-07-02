import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 定义克隆模型（使用与目标模型相同的架构）
def create_clone_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


# 加载已训练好的目标模型
def load_victim_model(model_path='victim_model.pth'):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    # 确保权重加载正确
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(f"成功加载目标模型权重从 {model_path}")
    except Exception as e:
        print(f"加载目标模型时出错: {e}")
        return None

    model = model.to(device)
    model.eval()  # 设为评估模式
    return model


# 定义DCGAN生成器和判别器
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        """
        参数：
            nz: 输入噪声向量的维度
            ngf: 生成器特征图的大小
            nc: 输出通道数
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入是nz维的随机噪声，输出是ngf*4的特征图
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 特征图大小: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 特征图大小: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 特征图大小: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 特征图大小: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 最终输出大小: (nc) x 64 x 64
        )
        # 对于CIFAR-10，我们需要调整输出大小
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)

    def forward(self, input):
        x = input.view(-1, input.size(1), 1, 1)  # 将输入重塑为4D张量
        x = self.main(x)
        return self.upsample(x)  # 调整为32x32大小


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        """
        参数：
            ndf: 判别器特征图的大小
            nc: 输入通道数
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入是(nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图大小: (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图大小: (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图大小: (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图大小: (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# 加载预训练的生成器
def load_generator(model_path='dcgan_generator_synthetic.pth', nc=1):
    nz = 100
    netG = Generator(nz=nz, nc=nc).to(device)

    try:
        netG.load_state_dict(torch.load(model_path))
        print(f"成功加载生成器权重从 {model_path}")
    except Exception as e:
        print(f"加载生成器时出错: {e}")
        return None

    netG.eval()
    return netG


# DFMS-HL算法主要实现
class DFMS_HL:
    def __init__(self, victim_model, generator, proxy_dataset=None, nc=3, nz=100, lambda_div=500):
        """
        初始化DFMS-HL算法

        参数:
            victim_model: 目标模型
            generator: 预训练的生成器
            nc: 图像通道数
            nz: 噪声向量维度
            lambda_div: 类多样性损失的权重系数
        """
        self.victim_model = victim_model
        self.victim_model.eval()  # 确保目标模型处于评估模式

        self.generator = generator
        self.generator.train()  # 生成器处于训练模式

        # 创建判别器
        self.discriminator = Discriminator(nc=nc).to(device)
        self.discriminator.apply(self._weights_init)

        # 创建克隆模型
        self.clone_model = create_clone_model().to(device)

        # 设置超参数
        self.nz = nz
        self.nc = nc
        self.lambda_div = lambda_div

        # 设置损失函数
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()

        # 查询计数器
        self.query_count = 0

        # 设置优化器 - 使用更合适的学习率和权重衰减
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_C = optim.SGD(self.clone_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler_C = CosineAnnealingLR(self.optimizer_C, T_max=200)

        # 最佳模型跟踪
        self.best_clone_acc = 0.0

        # 创建测试数据加载器用于验证
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=0)

        # 添加代理数据集
        self.proxy_dataset = proxy_dataset
        if proxy_dataset is not None:
            self.proxy_loader = DataLoader(proxy_dataset, batch_size=128, shuffle=True, num_workers=0)
            self.proxy_iterator = iter(self.proxy_loader)

    # 替换initialize_generator方法
    def initialize_generator(self, nG=2000, batch_size=128):
        """
        初始化生成器，迭代次数降低到合理水平
        论文可能指的是生成的样本总数而不是迭代次数
        """
        print("初始化生成器...")
        self.generator.train()
        self.discriminator.train()

        real_label = 1
        fake_label = 0

        # 预先准备代理数据批次（如果可用）
        proxy_batches = []
        if hasattr(self, 'proxy_loader') and self.proxy_loader is not None:
            for data, _ in self.proxy_loader:
                proxy_batches.append(data.to(device))
                if len(proxy_batches) >= 100:  # 缓存100个批次足够了
                    break

        pbar = tqdm(total=nG, desc="初始化生成器")

        for i in range(nG):
            # 生成随机噪声
            z = torch.randn(batch_size, self.nz, device=device)

            # 生成假图像
            fake_images = self.generator(z)
            fake_images = self._ensure_three_channels(fake_images)

            # 训练判别器
            self.optimizer_D.zero_grad()

            # 使用生成的假样本
            label = torch.full((batch_size,), fake_label, device=device, dtype=torch.float)
            output = self.discriminator(fake_images.detach())
            errD_fake = self.criterion_bce(output, label)
            errD_fake.backward()

            # 使用真实代理样本（从缓存获取）
            if proxy_batches:
                real_data = proxy_batches[i % len(proxy_batches)]
                if real_data.size(0) == batch_size:
                    label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
                    output = self.discriminator(real_data)
                    errD_real = self.criterion_bce(output, label)
                    errD_real.backward()
                    errD = errD_fake + errD_real
                else:
                    errD = errD_fake
            else:
                errD = errD_fake

            self.optimizer_D.step()

            # 训练生成器
            self.optimizer_G.zero_grad()

            # 对抗损失
            label.fill_(real_label)
            output = self.discriminator(fake_images)
            adv_loss = self.criterion_bce(output, label)

            # 类多样性损失
            with torch.no_grad():
                clone_outputs = self.clone_model(fake_images)

            div_loss = self.class_diversity_loss(clone_outputs)

            # 总生成器损失
            g_loss = adv_loss + self.lambda_div * div_loss
            g_loss.backward()
            self.optimizer_G.step()

            pbar.update(1)
            if (i + 1) % 100 == 0:
                pbar.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{errD.item():.4f}")

        pbar.close()
        print("生成器初始化完成")

    # 替换get_next_proxy_batch方法
    def get_next_proxy_batch(self, batch_size=128):
        """获取下一批代理数据，使用预先初始化的数据加载器"""
        if not hasattr(self, 'proxy_loader') or self.proxy_loader is None:
            return None

        # 如果没有初始化迭代器或者迭代器已经用尽
        if not hasattr(self, 'proxy_iterator') or self.proxy_iterator is None:
            self.proxy_iterator = iter(self.proxy_loader)

        try:
            data, _ = next(self.proxy_iterator)
            return data.to(device)
        except StopIteration:
            # 重新初始化迭代器
            self.proxy_iterator = iter(self.proxy_loader)
            try:
                data, _ = next(self.proxy_iterator)
                return data.to(device)
            except:
                return None

    def _weights_init(self, m):
        """权重初始化函数"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def _ensure_three_channels(self, x):
        """确保输入图像有3个通道"""
        if x.size(1) == 1:  # 如果是单通道图像
            return x.repeat(1, 3, 1, 1)  # 复制到三个通道
        return x

    def get_hard_label(self, x):
        """从目标模型获取硬标签"""
        with torch.no_grad():
            # 确保通道数正确
            x = self._ensure_three_channels(x)

            outputs = self.victim_model(x)
            _, predicted = outputs.max(1)
            self.query_count += x.size(0)  # 更新查询计数
        return predicted

    def evaluate_clone(self):
        """评估克隆模型在测试集上的准确率"""
        self.clone_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.clone_model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"克隆模型测试准确率: {accuracy:.2f}%")

        # 更新最佳模型
        if accuracy > self.best_clone_acc:
            self.best_clone_acc = accuracy
            torch.save(self.clone_model.state_dict(), 'best_clone_model_baseline.pth')
            print(f"保存新的最佳模型，准确率: {accuracy:.2f}%")

        return accuracy

    def initialize_clone(self, num_iterations=50000, batch_size=128, evaluate_every=10000):
        """初始化克隆模型，增加迭代次数和定期评估"""
        print("初始化克隆模型...")
        self.clone_model.train()

        # 创建一个进度条以便于跟踪
        pbar = tqdm(total=num_iterations, desc="初始化克隆模型")

        for i in range(num_iterations):
            # 生成随机噪声
            z = torch.randn(batch_size, self.nz, device=device)

            # 生成图像并确保通道数正确
            fake_images = self.generator(z)
            fake_images = self._ensure_three_channels(fake_images)

            # 从目标模型获取标签
            labels = self.get_hard_label(fake_images)

            # 训练克隆模型
            self.optimizer_C.zero_grad()
            outputs = self.clone_model(fake_images)
            loss = self.criterion_ce(outputs, labels)
            loss.backward()
            self.optimizer_C.step()
            self.scheduler_C.step()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", queries=self.query_count)

            # 定期评估
            if (i + 1) % evaluate_every == 0:
                acc = self.evaluate_clone()
                self.clone_model.train()  # 评估后切回训练模式

        pbar.close()
        print(f"克隆模型初始化完成，总查询数: {self.query_count}")

        # 最终评估
        final_acc = self.evaluate_clone()
        print(f"初始化后克隆模型准确率: {final_acc:.2f}%")

    # 替换class_diversity_loss方法
    def class_diversity_loss(self, batch_outputs):
        """计算类多样性损失 - 严格按照论文公式5"""
        batch_size = batch_outputs.size(0)
        softmax_outputs = torch.softmax(batch_outputs, dim=1)

        # 计算每个类别的平均置信度
        alpha_j = softmax_outputs.mean(dim=0)  # shape: [num_classes]

        # 计算负熵: Lclass_div = ∑(j=0 to K) [αj log αj]
        # 避免log(0)，添加小epsilon
        loss = torch.sum(alpha_j * torch.log(alpha_j + 1e-10))

        return loss  # 最小化负熵

    def visualize_class_distribution(self):
        """可视化生成器生成的图像的类别分布"""
        num_samples = 1000
        class_counts = torch.zeros(10, device=device)

        with torch.no_grad():
            for _ in range(10):  # 分批生成以避免内存问题
                z = torch.randn(100, self.nz, device=device)
                fake_images = self.generator(z)
                fake_images = self._ensure_three_channels(fake_images)
                labels = self.get_hard_label(fake_images)

                # 统计每个类别的数量
                for i in range(10):
                    class_counts[i] += (labels == i).sum().item()

        # 计算每个类别的百分比
        class_percentages = class_counts / class_counts.sum() * 100

        # 打印类别分布
        print("类别分布百分比:")
        for i in range(10):
            print(f"类别 {i}: {class_percentages[i].item():.2f}%")

        # 创建柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), class_percentages.cpu().numpy())
        plt.xlabel('类别')
        plt.ylabel('百分比 (%)')
        plt.title('生成图像的类别分布')
        plt.xticks(range(10))
        plt.savefig('class_distribution_baseline.png')
        plt.close()

    def train(self, num_queries=8000000, batch_size=128, g_steps=1, c_steps=1, evaluate_every=100000):
        """
        训练克隆模型，实现DFMS-HL算法

        参数:
            num_queries: 最大查询预算
            batch_size: 批量大小
            g_steps: 每次训练生成器的步数
            c_steps: 每次训练克隆模型的步数
            evaluate_every: 多少次查询后评估一次模型
        """
        print("开始DFMS-HL训练...")

        # 重置查询计数器
        initial_queries = self.query_count
        real_label = 1
        fake_label = 0

        # 保存训练历史
        history = {
            'g_loss': [],
            'div_loss': [],
            'adv_loss': [],
            'd_loss': [],
            'c_loss': [],
            'query_count': [],
            'accuracy': []
        }

        # 可视化初始类别分布
        print("初始类别分布:")
        self.visualize_class_distribution()

        # 创建一个进度条
        pbar = tqdm(total=num_queries, desc="DFMS-HL训练")
        pbar.update(self.query_count - initial_queries)

        # 训练循环，直到查询预算用完
        while self.query_count - initial_queries < num_queries:
            # 1. 训练生成器和判别器（固定克隆模型）
            self.generator.train()
            self.discriminator.train()
            self.clone_model.eval()

            g_loss_sum = 0
            d_loss_sum = 0
            div_loss_sum = 0
            adv_loss_sum = 0

            for _ in range(g_steps):
                # 生成随机噪声
                z = torch.randn(batch_size, self.nz, device=device)

                # 生成假图像
                fake_images = self.generator(z)
                fake_images = self._ensure_three_channels(fake_images)

                # 训练判别器
                self.optimizer_D.zero_grad()

                # 使用生成的假样本
                label = torch.full((batch_size,), fake_label, device=device, dtype=torch.float)
                output = self.discriminator(fake_images.detach())
                errD_fake = self.criterion_bce(output, label)
                errD_fake.backward()

                # 使用真实代理样本
                real_data = self.get_next_proxy_batch(batch_size)
                if real_data is not None and real_data.size(0) == batch_size:
                    label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
                    output = self.discriminator(real_data)
                    errD_real = self.criterion_bce(output, label)
                    errD_real.backward()
                    errD = errD_fake + errD_real
                else:
                    # 没有代理数据或批量大小不匹配时仅使用假样本
                    errD = errD_fake

                self.optimizer_D.step()
                d_loss_sum += errD.item()

                # 训练生成器
                self.optimizer_G.zero_grad()

                # 对抗损失
                label.fill_(real_label)
                output = self.discriminator(fake_images)
                adv_loss = self.criterion_bce(output, label)
                adv_loss_sum += adv_loss.item()

                # 类多样性损失
                with torch.no_grad():
                    clone_outputs = self.clone_model(fake_images)

                div_loss = self.class_diversity_loss(clone_outputs)
                div_loss_sum += div_loss.item()

                # 总生成器损失
                g_loss = adv_loss + self.lambda_div * div_loss
                g_loss_sum += g_loss.item()
                g_loss.backward()
                self.optimizer_G.step()

            # 2. 训练克隆模型（固定生成器和判别器）
            self.generator.eval()
            self.clone_model.train()

            c_loss_sum = 0
            for _ in range(c_steps):
                # 生成新的随机噪声
                z = torch.randn(batch_size, self.nz, device=device)

                # 生成图像
                with torch.no_grad():
                    fake_images = self.generator(z)
                    fake_images = self._ensure_three_channels(fake_images)

                # 从目标模型获取硬标签
                labels = self.get_hard_label(fake_images)

                # 训练克隆模型
                self.optimizer_C.zero_grad()
                outputs = self.clone_model(fake_images)
                loss = self.criterion_ce(outputs, labels)
                c_loss_sum += loss.item()
                loss.backward()
                self.optimizer_C.step()

            # 更新学习率
            self.scheduler_C.step()

            # 记录历史
            g_loss_avg = g_loss_sum / g_steps
            d_loss_avg = d_loss_sum / g_steps
            div_loss_avg = div_loss_sum / g_steps
            adv_loss_avg = adv_loss_sum / g_steps
            c_loss_avg = c_loss_sum / c_steps

            history['g_loss'].append(g_loss_avg)
            history['div_loss'].append(div_loss_avg)
            history['adv_loss'].append(adv_loss_avg)
            history['d_loss'].append(d_loss_avg)
            history['c_loss'].append(c_loss_avg)
            history['query_count'].append(self.query_count)

            # 更新进度条
            new_queries = self.query_count - initial_queries - pbar.n
            pbar.update(new_queries)
            pbar.set_postfix(g_loss=f"{g_loss_avg:.4f}", c_loss=f"{c_loss_avg:.4f}", queries=self.query_count)

            # 每evaluate_every次查询评估一次克隆模型性能
            if self.query_count - initial_queries >= len(history['accuracy']) * evaluate_every:
                acc = self.evaluate_clone()
                history['accuracy'].append(acc)

                # 可视化当前类别分布
                if len(history['accuracy']) % 10 == 0:
                    self.visualize_class_distribution()

                # 保存中间训练历史
                self._save_training_history(history)

                # 切回训练模式
                self.clone_model.train()

            # 检查是否超出查询预算
            if self.query_count - initial_queries >= num_queries:
                break

        pbar.close()

        # 保存最终模型和训练历史
        torch.save(self.clone_model.state_dict(), 'clone_model_final_baseline.pth')
        self._save_training_history(history)

        print(f"DFMS-HL训练完成，总查询数: {self.query_count}")
        print(f"最佳克隆模型准确率: {self.best_clone_acc:.2f}%")

        # 绘制训练损失
        self._plot_training_history(history)

        # 最终评估
        final_acc = self.evaluate_clone()
        print(f"最终克隆模型准确率: {final_acc:.2f}%")

        return history

    def _save_training_history(self, history):
        """保存训练历史到文件"""
        import pickle
        with open('training_history_baseline.pkl', 'wb') as f:
            pickle.dump(history, f)

    def _plot_training_history(self, history):
        """绘制训练历史图表"""
        plt.figure(figsize=(18, 12))

        # 绘制生成器损失
        plt.subplot(3, 2, 1)
        plt.plot(history['query_count'], history['g_loss'])
        plt.title('Generator Loss')
        plt.xlabel('Queries')

        # 绘制判别器损失
        plt.subplot(3, 2, 2)
        plt.plot(history['query_count'], history['d_loss'])
        plt.title('Discriminator Loss')
        plt.xlabel('Queries')

        # 绘制多样性损失
        plt.subplot(3, 2, 3)
        plt.plot(history['query_count'], history['div_loss'])
        plt.title('Diversity Loss')
        plt.xlabel('Queries')

        # 绘制对抗损失
        plt.subplot(3, 2, 4)
        plt.plot(history['query_count'], history['adv_loss'])
        plt.title('Adversarial Loss')
        plt.xlabel('Queries')

        # 绘制克隆损失
        plt.subplot(3, 2, 5)
        plt.plot(history['query_count'], history['c_loss'])
        plt.title('Clone Loss')
        plt.xlabel('Queries')

        # 绘制准确率
        evaluate_every = history['query_count'][-1] // len(history['accuracy'])
        query_points = [i * evaluate_every for i in range(len(history['accuracy']))]

        plt.subplot(3, 2, 6)
        plt.plot(query_points, history['accuracy'])
        plt.title('Clone Accuracy')
        plt.xlabel('Queries')
        plt.ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.savefig('dfms_hl_training_history_baseline.png')
        plt.close()


class SyntheticDataset(Dataset):
    """创建合成数据集，包含随机形状（三角形、矩形、椭圆或圆形）"""

    def __init__(self, size=50000, image_size=32):
        self.size = size
        self.image_size = image_size
        self.data = []

        # 生成合成数据
        for _ in range(size):
            img = self.generate_random_image()
            self.data.append(img)

    def generate_random_image(self):
        # 创建随机背景颜色
        bg_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

        # 创建图像
        img = Image.new('RGB', (self.image_size, self.image_size), bg_color)
        draw = ImageDraw.Draw(img)

        # 添加1-3个随机形状
        for _ in range(random.randint(1, 3)):
            # 选择形状类型
            shape_type = random.choice(['triangle', 'rectangle', 'ellipse', 'circle'])

            # 随机形状颜色
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

            # 随机位置和大小
            x1 = random.randint(0, self.image_size - 1)
            y1 = random.randint(0, self.image_size - 1)
            size = random.randint(5, 20)
            x2 = min(x1 + size, self.image_size - 1)
            y2 = min(y1 + size, self.image_size - 1)

            # 绘制形状
            if shape_type == 'triangle':
                # 三角形的三个点
                points = [
                    (x1, y1),
                    (x2, y1),
                    ((x1 + x2) // 2, y2)
                ]
                draw.polygon(points, fill=color)
            elif shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=color)
            elif shape_type == 'ellipse':
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:  # circle
                radius = size // 2
                center_x = x1 + radius
                center_y = y1 + radius
                draw.ellipse([center_x - radius, center_y - radius,
                              center_x + radius, center_y + radius], fill=color)

        # 转为灰度图（如论文所示）
        img = img.convert('L')
        # 转换为Tensor
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5,), (0.5,))(img)  # 标准化到[-1, 1]
        return img

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], 0  # 标签不重要，设为0


def get_proxy_dataset(num_classes=40):
    """获取CIFAR-100的子集作为代理数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化到[-1, 1]
    ])

    cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # 随机选择num_classes个类别
    selected_classes = random.sample(range(100), num_classes)
    indices = [i for i, (_, label) in enumerate(cifar100) if label in selected_classes]

    # 创建子集
    subset = torch.utils.data.Subset(cifar100, indices)
    return subset
# 主函数
if __name__ == "__main__":
    # 加载目标模型
    victim_model = load_victim_model('victim_model.pth')

    if victim_model is None:
        print("无法加载目标模型，请确保文件存在！")
        exit(1)

    # 选择使用哪种预训练生成器
    use_synthetic = False  # 设置为False以使用代理数据集的生成器

    if use_synthetic:
        # 使用合成数据
        proxy_dataset = SyntheticDataset(size=50000)
        generator = load_generator('dcgan_generator_synthetic.pth', nc=1)
        nc = 1
    else:
        # 使用CIFAR-100代理数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # 选择40个类别
        proxy_dataset = get_proxy_dataset(num_classes=40)
        generator = load_generator('dcgan_generator_proxy.pth', nc=3)
        nc = 3

    if generator is None:
        print("无法加载生成器模型，请确保文件存在！")
        exit(1)

    # 测试目标模型
    print("测试目标模型性能...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    victim_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = victim_model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    victim_acc = 100. * correct / total
    print(f"目标模型准确率: {victim_acc:.2f}%")

    # 初始化DFMS_HL
    dfms_hl = DFMS_HL(victim_model, generator, proxy_dataset=proxy_dataset, nc=nc, lambda_div=500)

    # 初始化克隆模型
    # dfms_hl.initialize_clone(num_iterations=20000, evaluate_every=20000)


    # 生成器初始化
    print("初始化生成器...")
    # dfms_hl.initialize_generator(nG=50000)  # 原论文中的nG迭代次数


    # 主训练过程
    history = dfms_hl.train(
        num_queries=8000000,  # 论文中提到的800万查询预算
        batch_size=128,
        g_steps=1,  # 每次迭代训练生成器1次
        c_steps=1,  # 每次迭代训练克隆模型1次，确保严格交替
        evaluate_every=100000  # 每10万次查询评估一次
    )
