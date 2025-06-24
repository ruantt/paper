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
class DFMS_HL_Swift(nn.Module):
    def __init__(self, victim_model, generator, proxy_dataset=None, nc=3, nz=100,
                 lambda_div=500, lambda_contrast=1.0, temperature=0.07,
                 feature_dim=128, queue_size=4096):
        """
        初始化增强版DFMS-HL算法，整合SwiftThief的对比学习

        参数:
            victim_model: 目标模型
            generator: 预训练的生成器
            proxy_dataset: 代理数据集
            nc: 图像通道数
            nz: 噪声向量维度
            lambda_div: 类多样性损失的权重系数
            lambda_contrast: 对比学习损失的权重系数
            temperature: 对比学习温度参数
            feature_dim: 特征维度
            queue_size: 特征队列大小，用于对比学习
        """
        # 继承nn.Module的初始化
        super(DFMS_HL_Swift, self).__init__()

        # 保留原始DFMS-HL的初始化代码
        self.victim_model = victim_model
        self.victim_model.eval()

        self.generator = generator
        self.generator.train()

        # 创建判别器
        self.discriminator = Discriminator(nc=nc).to(device)
        self.discriminator.apply(self._weights_init)

        # 创建克隆模型
        self.clone_model = create_clone_model().to(device)

        # 添加特征提取器，用于对比学习
        self.feature_extractor = self._create_feature_extractor().to(device)

        # 设置超参数
        self.nz = nz
        self.nc = nc
        self.lambda_div = lambda_div
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature

        # 设置损失函数
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()

        # 查询计数器
        self.query_count = 0

        # 设置优化器
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_C = optim.SGD(self.clone_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.optimizer_F = optim.SGD(self.feature_extractor.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler_C = CosineAnnealingLR(self.optimizer_C, T_max=200)
        self.scheduler_F = CosineAnnealingLR(self.optimizer_F, T_max=200)

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

        # 为对比学习创建特征队列
        self.register_buffer("queue", torch.randn(queue_size, feature_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

        # 类别查询计数，用于优先级采样
        self.register_buffer("class_query_counts", torch.zeros(10))  # 假设有10个类别

        # 类别查询计数，用于优先级采样
        self.class_query_counts = torch.zeros(10, device=device)  # 假设有10个类别

    def _create_feature_extractor(self):
        """创建用于对比学习的特征提取器"""
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        # 移除最后的全连接层，添加投影头
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, 128)  # 投影到128维特征空间
        )
        return model

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
            x = self._ensure_three_channels(x)
            outputs = self.victim_model(x)
            _, predicted = outputs.max(1)
            self.query_count += x.size(0)  # 更新查询计数

            # 更新类别查询计数
            for cls_idx in range(10):
                self.class_query_counts[cls_idx] += (predicted == cls_idx).sum().item()

        return predicted

    def get_soft_label(self, x):
        """从目标模型获取软标签(概率分布)"""
        with torch.no_grad():
            x = self._ensure_three_channels(x)
            outputs = self.victim_model(x)
            probs = torch.softmax(outputs, dim=1)
            self.query_count += x.size(0)  # 更新查询计数

            # 更新类别查询计数
            _, predicted = outputs.max(1)
            for cls_idx in range(10):
                self.class_query_counts[cls_idx] += (predicted == cls_idx).sum().item()

        return probs

    def class_diversity_loss(self, batch_outputs):
        """计算类多样性损失"""
        softmax_outputs = torch.softmax(batch_outputs, dim=1)
        alpha_j = softmax_outputs.mean(dim=0)
        loss = torch.sum(alpha_j * torch.log(alpha_j + 1e-10))
        return loss

    def _dequeue_and_enqueue(self, keys):
        """更新特征队列，用于对比学习"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 替换队列中的键
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # 处理环形队列边界
            first_part = self.queue_size - ptr
            second_part = batch_size - first_part
            self.queue[ptr:] = keys[:first_part]
            self.queue[:second_part] = keys[first_part:]

        # 更新指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def contrastive_loss(self, features, labels=None):
        """
        计算对比学习损失
        """
        features = nn.functional.normalize(features, dim=1)
        batch_size = features.size(0)

        # 计算特征之间的相似度
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建标签掩码，用于排除自身
        mask = torch.eye(batch_size, device=features.device)

        if labels is not None:
            # 监督对比学习 - 相同类别的样本被视为正样本
            pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
            # 排除对角线(自身)
            pos_mask = pos_mask - mask
            # 确保每个样本至少有一个正样本(避免除零错误)
            pos_samples = torch.maximum(pos_mask.sum(1), torch.ones(batch_size, device=features.device))

            # 计算正样本对的损失
            log_prob = sim_matrix - torch.log(torch.exp(sim_matrix).sum(1, keepdim=True) + 1e-10)

            # 计算每个锚点的监督对比损失
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_samples
            loss = -mean_log_prob_pos.mean()
        else:
            # 自监督对比学习
            log_prob = sim_matrix - torch.log(torch.exp(sim_matrix).sum(1, keepdim=True) + 1e-10)

            # 对角线掩码，排除自身
            neg_mask = 1.0 - mask

            # 计算对比损失
            mean_log_prob_neg = (neg_mask * log_prob).sum(1) / (batch_size - 1)
            loss = -mean_log_prob_neg.mean()

        return loss

    def priority_sampling(self, batch_size):
        """
        基于类别频率的优先级采样
        较少查询的类别获得更高的采样优先级
        """
        # 添加平滑因子，避免除零错误
        smooth_counts = self.class_query_counts + 1.0

        # 计算反比例采样权重
        weights = 1.0 / smooth_counts
        weights = weights / weights.sum()

        # 根据权重对类别进行采样
        sampled_classes = torch.multinomial(weights, batch_size, replacement=True)

        # 为每个采样的类别生成噪声
        z = torch.randn(batch_size, self.nz, device=device)

        return z, sampled_classes


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
        plt.savefig('class_distribution.png')
        plt.close()

    def get_next_proxy_batch(self):
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
            torch.save(self.clone_model.state_dict(), 'best_clone_model.pth')
            print(f"保存新的最佳模型，准确率: {accuracy:.2f}%")

        return accuracy

    def train(self, num_queries=8000000, batch_size=128, g_steps=1, c_steps=1, evaluate_every=100000):
        """
        训练克隆模型，实现增强版DFMS-HL算法
        """
        print("开始SwiftThief增强版DFMS-HL训练...")

        # 重置查询计数器
        initial_queries = self.query_count
        real_label = 1
        fake_label = 0

        # 保存训练历史
        history = {
            'g_loss': [], 'div_loss': [], 'adv_loss': [], 'd_loss': [],
            'c_loss': [], 'contrast_loss': [], 'query_count': [], 'accuracy': []
        }

        # 可视化初始类别分布
        print("初始类别分布:")
        self.visualize_class_distribution()

        # 创建一个进度条
        pbar = tqdm(total=num_queries, desc="SwiftThief-DFMS-HL训练")
        pbar.update(self.query_count - initial_queries)

        # 训练循环，直到查询预算用完
        while self.query_count - initial_queries < num_queries:
            # 1. 训练生成器和判别器（固定克隆模型）
            self.generator.train()
            self.discriminator.train()
            self.clone_model.eval()
            self.feature_extractor.eval()

            g_loss_sum = 0
            d_loss_sum = 0
            div_loss_sum = 0
            adv_loss_sum = 0

            for _ in range(g_steps):
                # 生成随机噪声，使用优先级采样
                z, target_classes = self.priority_sampling(batch_size)

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
                real_data = self.get_next_proxy_batch()
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

            # 2. 训练克隆模型和特征提取器（固定生成器和判别器）
            self.generator.eval()
            self.clone_model.train()
            self.feature_extractor.train()

            c_loss_sum = 0
            contrast_loss_sum = 0

            for _ in range(c_steps):
                # 生成新的随机噪声，使用优先级采样
                z, target_classes = self.priority_sampling(batch_size)

                # 生成图像
                with torch.no_grad():
                    fake_images = self.generator(z)
                    fake_images = self._ensure_three_channels(fake_images)

                # 从目标模型获取软标签
                soft_labels = self.get_soft_label(fake_images)
                hard_labels = soft_labels.argmax(dim=1)

                # 提取特征
                self.optimizer_F.zero_grad()
                features = self.feature_extractor(fake_images)

                # 计算监督对比学习损失
                contrast_loss = self.contrastive_loss(features, hard_labels)
                contrast_loss_sum += contrast_loss.item()

                # 提取未查询数据的特征（如果有代理数据集）
                if self.proxy_dataset is not None:
                    proxy_data = self.get_next_proxy_batch()
                    if proxy_data is not None and proxy_data.size(0) == batch_size:
                        proxy_features = self.feature_extractor(proxy_data)
                        # 自监督对比学习损失
                        unsup_contrast_loss = self.contrastive_loss(proxy_features)
                        contrast_loss += unsup_contrast_loss
                        contrast_loss_sum += unsup_contrast_loss.item()

                # 训练克隆模型
                self.optimizer_C.zero_grad()
                outputs = self.clone_model(fake_images)

                # 使用软标签进行蒸馏
                kl_loss = nn.KLDivLoss(reduction='batchmean')(
                    nn.functional.log_softmax(outputs, dim=1),
                    soft_labels
                )

                # 总损失 = KL损失 + 对比学习损失
                total_loss = kl_loss + self.lambda_contrast * contrast_loss
                c_loss_sum += kl_loss.item()

                total_loss.backward()
                self.optimizer_C.step()
                self.optimizer_F.step()

                # 更新特征队列
                with torch.no_grad():
                    keys = self.feature_extractor(fake_images)
                    keys = nn.functional.normalize(keys, dim=1)
                    self._dequeue_and_enqueue(keys.detach())

            # 更新学习率
            self.scheduler_C.step()
            self.scheduler_F.step()

            # 记录历史
            g_loss_avg = g_loss_sum / g_steps
            d_loss_avg = d_loss_sum / g_steps
            div_loss_avg = div_loss_sum / g_steps
            adv_loss_avg = adv_loss_sum / g_steps
            c_loss_avg = c_loss_sum / c_steps
            contrast_loss_avg = contrast_loss_sum / c_steps

            history['g_loss'].append(g_loss_avg)
            history['div_loss'].append(div_loss_avg)
            history['adv_loss'].append(adv_loss_avg)
            history['d_loss'].append(d_loss_avg)
            history['c_loss'].append(c_loss_avg)
            history['contrast_loss'].append(contrast_loss_avg)
            history['query_count'].append(self.query_count)

            # 更新进度条
            new_queries = self.query_count - initial_queries - pbar.n
            pbar.update(new_queries)
            pbar.set_postfix(g_loss=f"{g_loss_avg:.4f}", c_loss=f"{c_loss_avg:.4f}",
                             contrast=f"{contrast_loss_avg:.4f}", queries=self.query_count)

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
                self.feature_extractor.train()

            # 检查是否超出查询预算
            if self.query_count - initial_queries >= num_queries:
                break

        pbar.close()

        # 保存最终模型和训练历史
        torch.save(self.clone_model.state_dict(), 'clone_model_final.pth')
        torch.save(self.feature_extractor.state_dict(), 'feature_extractor_final.pth')
        self._save_training_history(history)

        print(f"SwiftThief-DFMS-HL训练完成，总查询数: {self.query_count}")
        print(f"最佳克隆模型准确率: {self.best_clone_acc:.2f}%")

        # 绘制训练损失
        self._plot_training_history(history)

        # 最终评估
        final_acc = self.evaluate_clone()
        print(f"最终克隆模型准确率: {final_acc:.2f}%")

        return history

    # 初始化克隆模型，使用对比学习
    def initialize_clone_with_contrastive(self, num_iterations=50000, batch_size=128, evaluate_every=10000):
        """初始化克隆模型，使用对比学习"""
        print("初始化克隆模型...")
        self.clone_model.train()
        self.feature_extractor.train()

        # 创建一个进度条以便于跟踪
        pbar = tqdm(total=num_iterations, desc="初始化克隆模型")

        for i in range(num_iterations):
            # 生成随机噪声，使用优先级采样
            z, target_classes = self.priority_sampling(batch_size)

            # 生成图像并确保通道数正确
            fake_images = self.generator(z)
            fake_images = self._ensure_three_channels(fake_images)

            # 从目标模型获取标签
            hard_labels = self.get_hard_label(fake_images)

            # 提取特征
            self.optimizer_F.zero_grad()
            features = self.feature_extractor(fake_images)

            # 计算监督对比学习损失
            contrast_loss = self.contrastive_loss(features, hard_labels)

            # 训练克隆模型
            self.optimizer_C.zero_grad()
            outputs = self.clone_model(fake_images)
            ce_loss = self.criterion_ce(outputs, hard_labels)

            # 总损失 = 交叉熵损失 + 对比学习损失
            total_loss = ce_loss + self.lambda_contrast * contrast_loss

            total_loss.backward()
            self.optimizer_C.step()
            self.optimizer_F.step()
            self.scheduler_C.step()
            self.scheduler_F.step()

            # 更新特征队列
            with torch.no_grad():
                keys = self.feature_extractor(fake_images)
                keys = nn.functional.normalize(keys, dim=1)
                self._dequeue_and_enqueue(keys.detach())

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", contrast=f"{contrast_loss.item():.4f}",
                             queries=self.query_count)

            # 定期评估
            if (i + 1) % evaluate_every == 0:
                acc = self.evaluate_clone()
                self.clone_model.train()  # 评估后切回训练模式
                self.feature_extractor.train()

        pbar.close()
        print(f"克隆模型初始化完成，总查询数: {self.query_count}")

        # 最终评估
        final_acc = self.evaluate_clone()
        print(f"初始化后克隆模型准确率: {final_acc:.2f}%")

    def _save_training_history(self, history):
        """保存训练历史到文件"""
        import pickle
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history, f)

    def _plot_training_history(self, history):
        """绘制训练历史图表"""
        plt.figure(figsize=(18, 15))

        # 绘制生成器损失
        plt.subplot(4, 2, 1)
        plt.plot(history['query_count'], history['g_loss'])
        plt.title('Generator Loss')
        plt.xlabel('Queries')

        # 绘制判别器损失
        plt.subplot(4, 2, 2)
        plt.plot(history['query_count'], history['d_loss'])
        plt.title('Discriminator Loss')
        plt.xlabel('Queries')

        # 绘制多样性损失
        plt.subplot(4, 2, 3)
        plt.plot(history['query_count'], history['div_loss'])
        plt.title('Diversity Loss')
        plt.xlabel('Queries')

        # 绘制对抗损失
        plt.subplot(4, 2, 4)
        plt.plot(history['query_count'], history['adv_loss'])
        plt.title('Adversarial Loss')
        plt.xlabel('Queries')

        # 绘制MMD损失
        plt.subplot(4, 2, 5)
        plt.plot(history['query_count'], history['mmd_loss'])
        plt.title('MMD Loss')
        plt.xlabel('Queries')

        # 绘制克隆损失
        plt.subplot(4, 2, 6)
        plt.plot(history['query_count'], history['c_loss'])
        plt.title('Clone Loss')
        plt.xlabel('Queries')

        # 绘制准确率
        evaluate_every = history['query_count'][-1] // len(history['accuracy']) if history['accuracy'] else 1
        query_points = [i * evaluate_every for i in range(len(history['accuracy']))]

        plt.subplot(4, 2, 7)
        plt.plot(query_points, history['accuracy'])
        plt.title('Clone Accuracy')
        plt.xlabel('Queries')
        plt.ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.savefig('active_dfms_hl_training_history.png')
        plt.close()
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
    dfms_hl = DFMS_HL_Swift(victim_model, generator, proxy_dataset=proxy_dataset, nc=nc, lambda_div=500)

    # 初始化克隆模型
    dfms_hl.initialize_clone(num_iterations=20000, evaluate_every=20000)


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

    # 跑一次 ，有可能
    # 那个是生成的数据
    # 可以的 可以的，你可以看下数据迭代过程