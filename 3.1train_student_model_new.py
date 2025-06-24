import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import torch.nn.functional as F

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


# 定义一个特征提取器，用于MMD计算
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # 使用ResNet的前几层作为特征提取器
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


# 批量主动学习的DFMS-HL算法
class DFMS_HL_Active:
    def __init__(self, victim_model, generator, proxy_dataset=None, nc=3, nz=100, lambda_div=500,
                 batch_size=128, active_strategy='uncertainty', k=10, alpha=0.5):
        """
        初始化带有主动学习的DFMS-HL算法

        参数:
            victim_model: 目标模型
            generator: 预训练的生成器
            proxy_dataset: 代理数据集
            nc: 图像通道数
            nz: 噪声向量维度
            lambda_div: 类多样性损失的权重系数
            batch_size: 批量大小
            active_strategy: 主动学习策略 ('uncertainty', 'boundary', 'cluster', 'combined')
            k: 聚类数量
            alpha: 组合策略的权重参数
        """
        self.current_iteration = 0
        self.temperature = 1.0
        self.min_temp = 0.5
        self.max_temp = 2.0
        self.cycle_length = 200000

        self.victim_model = victim_model
        self.victim_model.eval()  # 确保目标模型处于评估模式

        self.generator = generator
        self.generator.train()  # 生成器处于训练模式

        # 创建判别器
        self.discriminator = Discriminator(nc=nc).to(device)
        self.discriminator.apply(self._weights_init)

        # 创建克隆模型
        self.clone_model = create_clone_model().to(device)

        # 创建特征提取器，用于MMD计算
        self.feature_extractor = FeatureExtractor(self.victim_model).to(device)
        self.feature_extractor.eval()

        # 设置超参数
        self.nz = nz
        self.nc = nc
        self.lambda_div = lambda_div
        self.batch_size = batch_size
        self.active_strategy = active_strategy
        self.k = k
        self.alpha = alpha

        # 保存噪声向量池，用于优化选择
        self.noise_pool_size = batch_size * 10
        self.noise_pool = torch.randn(self.noise_pool_size, self.nz, device=device)

        # 设置损失函数
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()

        # 查询计数器
        self.query_count = 0

        # 设置优化器
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
            self.proxy_loader = DataLoader(proxy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            self.proxy_iterator = iter(self.proxy_loader)

    def anneal_temperature(self):
        """周期性温度退火，促进探索与利用的平衡"""
        cycle = (self.current_iteration % self.cycle_length) / self.cycle_length
        # 创建周期性的温度变化，在探索和利用之间切换
        if cycle < 0.5:
            # 降温阶段 - 增强利用
            temp = self.max_temp - (self.max_temp - self.min_temp) * (cycle * 2)
        else:
            # 升温阶段 - 增强探索
            temp = self.min_temp + (self.max_temp - self.min_temp) * ((cycle - 0.5) * 2)

        self.temperature = temp
        return temp

    def adjust_diversity_weight(self, class_distribution):
        """基于当前类别分布动态调整多样性损失权重，添加数值稳定性保护"""
        # 计算当前类别分布的熵
        distribution = class_distribution / (class_distribution.sum() + 1e-7)
        entropy = -torch.sum(distribution * torch.log(distribution + 1e-7))
        max_entropy = torch.log(torch.tensor(10.0, device=device))  # 理想均匀分布的熵

        # 根据分布接近均匀程度调整权重
        normalized_entropy = torch.clamp(entropy / max_entropy, 0.2, 1.0)

        # 使用更安全的线性调整替代指数调整
        base_weight = self.lambda_div
        # 线性调整: 较低熵值对应较高权重，但有上限
        adjusted_weight = base_weight * (2.0 - normalized_entropy)
        # 确保权重在合理范围内
        adjusted_weight = torch.clamp(adjusted_weight, base_weight * 0.5, 2)

        # 添加额外保护，确保返回合理值
        if torch.isnan(adjusted_weight) or torch.isinf(adjusted_weight):
            print("警告: 检测到调整的多样性权重为NaN或Inf，使用基础权重")
            return torch.tensor(base_weight, device=device)

        return adjusted_weight

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

    def compute_mmd(self, x, y):
        """计算最大平均差异(MMD)，添加数值稳定性保护"""
        # 确保输入有效且大小匹配
        if x.size(0) == 0 or y.size(0) == 0:
            return torch.tensor(0.0, device=device)

        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * xy

        # 添加数值稳定性，避免极大值
        dxx = torch.clamp(dxx, 0.0, 1000.0)
        dyy = torch.clamp(dyy, 0.0, 1000.0)
        dxy = torch.clamp(dxy, 0.0, 1000.0)

        XX = torch.zeros(xx.shape).to(device)
        YY = torch.zeros(xx.shape).to(device)
        XY = torch.zeros(xy.shape).to(device)

        # 使用更安全的带宽范围
        bandwidth_range = [0.5, 1.0, 2.0, 5.0]
        for a in bandwidth_range:
            XX += torch.exp(torch.clamp(-0.5 * dxx / a, -50.0, 50.0))
            YY += torch.exp(torch.clamp(-0.5 * dyy / a, -50.0, 50.0))
            XY += torch.exp(torch.clamp(-0.5 * dxy / a, -50.0, 50.0))

        # 计算最终结果，添加数值稳定性检查
        result = torch.mean(XX) + torch.mean(YY) - 2. * torch.mean(XY)

        # 确保结果有效
        if torch.isnan(result) or torch.isinf(result):
            print("警告: MMD计算得到NaN或Inf值，返回默认值")
            return torch.tensor(0.01, device=device)

        # 限制结果在合理范围内
        return torch.clamp(result, 0.0, 100.0)

    def calculate_uncertainty(self, outputs):
        """
        计算预测的不确定性

        参数:
            outputs: 模型输出的logits

        返回:
            uncertainty: 基于熵的不确定性得分
        """
        probs = F.softmax(outputs, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy

    def calculate_boundary_score(self, outputs):
        """
        计算样本到决策边界的距离

        参数:
            outputs: 模型输出的logits

        返回:
            boundary_score: 边界分数（越小表示越接近边界）
        """
        sorted_outputs, _ = torch.sort(outputs, dim=1, descending=True)
        # 最高类别和次高类别之间的差距越小，越接近决策边界
        margin = sorted_outputs[:, 0] - sorted_outputs[:, 1]
        return margin

    def select_batch_uncertainty(self, candidate_images, candidate_outputs, batch_size):
        """
        基于不确定性选择批次

        参数:
            candidate_images: 候选图像
            candidate_outputs: 候选图像的模型输出
            batch_size: 批量大小

        返回:
            selected_images: 选择的图像批次
            selected_indices: 选择的索引
        """
        uncertainty = self.calculate_uncertainty(candidate_outputs)

        # 选择不确定性最高的样本
        _, indices = torch.sort(uncertainty, descending=True)
        selected_indices = indices[:batch_size]
        selected_images = candidate_images[selected_indices]

        return selected_images, selected_indices

    def select_batch_boundary(self, candidate_images, candidate_outputs, batch_size):
        """
        基于决策边界选择批次

        参数:
            candidate_images: 候选图像
            candidate_outputs: 候选图像的模型输出
            batch_size: 批量大小

        返回:
            selected_images: 选择的图像批次
            selected_indices: 选择的索引
        """
        boundary_score = self.calculate_boundary_score(candidate_outputs)

        # 选择边界分数最低的样本（最接近决策边界）
        _, indices = torch.sort(boundary_score)
        selected_indices = indices[:batch_size]
        selected_images = candidate_images[selected_indices]

        return selected_images, selected_indices

    def select_batch_cluster(self, candidate_images, batch_size):
        """
        基于聚类选择多样性批次

        参数:
            candidate_images: 候选图像
            batch_size: 批量大小

        返回:
            selected_images: 选择的图像批次
            selected_indices: 选择的索引
        """
        # 提取特征
        with torch.no_grad():
            features = self.feature_extractor(self._ensure_three_channels(candidate_images))

        # 使用CPU上的KMeans进行聚类
        features_np = features.cpu().numpy()
        kmeans = KMeans(n_clusters=min(self.k, len(features_np)), random_state=0, n_init='auto').fit(features_np)
        centers = torch.tensor(kmeans.cluster_centers_, device=device)

        # 计算每个样本到最近的聚类中心的距离
        selected_indices = []
        remaining_indices = list(range(len(features)))

        # 为每个聚类选择一个代表性样本
        for i in range(min(self.k, batch_size)):
            cluster_indices = [idx for idx in remaining_indices if kmeans.labels_[idx] == i]
            if not cluster_indices:
                continue

            # 找到离中心最近的样本
            cluster_features = features[cluster_indices]
            center = centers[i].unsqueeze(0)
            distances = torch.cdist(cluster_features, center).squeeze()
            closest_idx = cluster_indices[torch.argmin(distances).item()]

            selected_indices.append(closest_idx)
            remaining_indices = [idx for idx in remaining_indices if idx != closest_idx]

        # 如果选择的样本数小于batch_size，则随机填充
        if len(selected_indices) < batch_size:
            remaining_to_select = batch_size - len(selected_indices)
            if remaining_indices and remaining_to_select > 0:
                additional_indices = random.sample(remaining_indices, min(remaining_to_select, len(remaining_indices)))
                selected_indices.extend(additional_indices)

        selected_indices = torch.tensor(selected_indices, device=device)
        selected_images = candidate_images[selected_indices]

        return selected_images, selected_indices

    def select_batch_balanced(self, candidate_images, candidate_outputs, batch_size):
        """基于类别平衡的批量选择策略"""
        # 获取预测类别
        _, predicted_classes = torch.max(candidate_outputs, 1)

        # 计算当前类别分布
        class_counts = torch.zeros(10, device=device)
        for i in range(10):
            class_counts[i] = (predicted_classes == i).sum().item()

        # 计算理想采样概率（类别越少权重越高）
        epsilon = 1e-5  # 避免除零
        sampling_weights = 1.0 / (class_counts + epsilon)
        sampling_weights = sampling_weights / sampling_weights.sum()

        # 为每个样本分配优先级（稀有类别优先）
        sample_priorities = torch.zeros(len(candidate_images), device=device)
        for i in range(10):
            class_mask = (predicted_classes == i)
            sample_priorities[class_mask] = sampling_weights[i]

        # 添加随机噪声以打破同类样本的排序
        sample_priorities += torch.rand_like(sample_priorities) * 0.1

        # 以加权方式采样
        _, indices = torch.sort(sample_priorities, descending=True)
        selected_indices = indices[:batch_size]

        return candidate_images[selected_indices], selected_indices

    def gradient_guided_sampling(self, candidate_images, candidate_outputs, labels, batch_size):
        """基于预测与真实标签间差异的梯度大小选择信息量最大的样本"""
        # 计算每个样本的损失值作为信息量的代理
        losses = []

        for i, (img, output, label) in enumerate(zip(candidate_images, candidate_outputs, labels)):
            loss = F.cross_entropy(output.unsqueeze(0), label.unsqueeze(0))
            losses.append(loss.item())

        # 将损失转换为张量
        losses = torch.tensor(losses, device=device)

        # 选择损失最大的样本（最具信息量）
        _, indices = torch.sort(losses, descending=True)
        selected_indices = indices[:batch_size]

        return candidate_images[selected_indices], selected_indices

    def select_active_batch(self, batch_size=None):
        """
        使用混合采样策略选择批次

        参数:
            batch_size: 批量大小（默认使用初始化时的batch_size）

        返回:
            selected_images: 选择的图像批次
            selected_labels: 对应的标签
        """
        if batch_size is None:
            batch_size = self.batch_size

        # 更新迭代计数并调整温度
        self.current_iteration += 1
        self.anneal_temperature()

        # 生成候选图像池（扩大候选池大小）
        pool_size = self.noise_pool_size
        with torch.no_grad():
            candidate_images = self.generator(self.noise_pool)
            candidate_images = self._ensure_three_channels(candidate_images)

            # 获取克隆模型对候选图像的预测
            clone_outputs = self.clone_model(candidate_images)

            # 获取目标模型的标签（仅获取一次以减少查询数）
            labels = self.get_hard_label(candidate_images)

        # 计算当前的类别分布
        class_counts = torch.zeros(10, device=device)
        for i in range(10):
            class_counts[i] = (labels == i).sum().item()

        # 动态调整类多样性损失权重
        adjusted_div_weight = self.adjust_diversity_weight(class_counts)

        # 根据类别分布情况调整各策略的比例
        # 如果存在缺失类别或严重不平衡，增加类别平衡采样的比例
        missing_classes = (class_counts == 0).sum().item()
        class_imbalance = torch.std(class_counts) / torch.mean(class_counts)

        # 根据类别不平衡程度动态调整各种策略的比例
        if missing_classes > 0 or class_imbalance > 1.0:
            # 类别严重不平衡情况
            balanced_ratio = min(0.5 + missing_classes * 0.1, 0.7)  # 最高70%
            uncertainty_ratio = 0.1
            boundary_ratio = 0.1
            cluster_ratio = 0.1
            gradient_ratio = 1.0 - balanced_ratio - uncertainty_ratio - boundary_ratio - cluster_ratio
        else:
            # 相对平衡情况
            balanced_ratio = max(0.2, 1.0 - (1.0 / class_imbalance))
            uncertainty_ratio = 0.2
            boundary_ratio = 0.2
            cluster_ratio = 0.2
            gradient_ratio = 1.0 - balanced_ratio - uncertainty_ratio - boundary_ratio - cluster_ratio

        # 确保比例之和为1且都为正数
        total_ratio = balanced_ratio + uncertainty_ratio + boundary_ratio + cluster_ratio + gradient_ratio
        balanced_ratio /= total_ratio
        uncertainty_ratio /= total_ratio
        boundary_ratio /= total_ratio
        cluster_ratio /= total_ratio
        gradient_ratio /= total_ratio

        # 计算每种策略的样本数
        balanced_count = max(1, int(batch_size * balanced_ratio))
        uncertainty_count = max(1, int(batch_size * uncertainty_ratio))
        boundary_count = max(1, int(batch_size * boundary_ratio))
        cluster_count = max(1, int(batch_size * cluster_ratio))
        gradient_count = batch_size - balanced_count - uncertainty_count - boundary_count - cluster_count

        # 应用温度缩放到softmax输出
        scaled_outputs = clone_outputs / self.temperature

        # 使用各种策略选择样本
        balanced_images, balanced_indices = self.select_batch_balanced(
            candidate_images, scaled_outputs, balanced_count)

        # 从剩余的候选中选择其他策略样本
        remaining_mask = torch.ones(len(candidate_images), dtype=torch.bool, device=device)
        remaining_mask[balanced_indices] = False

        remaining_images = candidate_images[remaining_mask]
        remaining_outputs = scaled_outputs[remaining_mask]
        remaining_labels = labels[remaining_mask]

        # 如果剩余样本不足，调整计数
        remaining_count = remaining_images.size(0)
        if remaining_count < uncertainty_count + boundary_count + cluster_count + gradient_count:
            total = uncertainty_count + boundary_count + cluster_count + gradient_count
            uncertainty_count = max(1, int(remaining_count * uncertainty_count / total))
            boundary_count = max(1, int(remaining_count * boundary_count / total))
            cluster_count = max(1, int(remaining_count * cluster_count / total))
            gradient_count = remaining_count - uncertainty_count - boundary_count - cluster_count

        # 选择不确定性样本
        uncertainty_images, uncertainty_indices = self.select_batch_uncertainty(
            remaining_images, remaining_outputs, uncertainty_count)

        # 更新剩余掩码
        remaining_mask = torch.ones(len(remaining_images), dtype=torch.bool, device=device)
        remaining_mask[uncertainty_indices] = False

        remaining_images = remaining_images[remaining_mask]
        remaining_outputs = remaining_outputs[remaining_mask]
        remaining_labels = remaining_labels[remaining_mask]

        # 选择边界样本
        boundary_images, boundary_indices = self.select_batch_boundary(
            remaining_images, remaining_outputs, boundary_count)

        # 更新剩余掩码
        remaining_mask = torch.ones(len(remaining_images), dtype=torch.bool, device=device)
        remaining_mask[boundary_indices] = False

        remaining_images = remaining_images[remaining_mask]
        remaining_outputs = remaining_outputs[remaining_mask]
        remaining_labels = remaining_labels[remaining_mask]

        # 选择聚类样本
        cluster_images, cluster_indices = self.select_batch_cluster(
            remaining_images, cluster_count)

        # 更新剩余掩码
        remaining_mask = torch.ones(len(remaining_images), dtype=torch.bool, device=device)
        remaining_mask[cluster_indices] = False

        remaining_images = remaining_images[remaining_mask]
        remaining_outputs = remaining_outputs[remaining_mask]
        remaining_labels = remaining_labels[remaining_mask]

        # 选择梯度样本
        if gradient_count > 0 and len(remaining_images) > 0:
            gradient_images, _ = self.gradient_guided_sampling(
                remaining_images, remaining_outputs, remaining_labels, gradient_count)
        else:
            gradient_images = torch.tensor([], device=device)

        # 合并所有选择的样本
        selected_indices = []

        if len(balanced_images) > 0:
            selected_indices.append(balanced_images)
        if len(uncertainty_images) > 0:
            selected_indices.append(uncertainty_images)
        if len(boundary_images) > 0:
            selected_indices.append(boundary_images)
        if len(cluster_images) > 0:
            selected_indices.append(cluster_images)
        if len(gradient_images) > 0:
            selected_indices.append(gradient_images)

        if not selected_indices:
            # 如果所有策略都失败，回退到随机选择
            selected_indices = torch.randperm(len(candidate_images), device=device)[:batch_size]
            selected_images = candidate_images[selected_indices]
            selected_labels = labels[selected_indices]
        else:
            # 合并所有选择的样本
            selected_images = torch.cat(selected_indices, dim=0)
            # 如果选择的样本数超过批量大小，随机裁剪
            if selected_images.size(0) > batch_size:
                perm = torch.randperm(selected_images.size(0), device=device)
                selected_images = selected_images[perm[:batch_size]]

            # 获取最终标签
            selected_labels = self.get_hard_label(selected_images)

        # 更新噪声池
        self.noise_pool = torch.randn(self.noise_pool_size, self.nz, device=device)

        # 更新lambda_div（类多样性损失权重）
        self.lambda_div = adjusted_div_weight.item()

        return selected_images, selected_labels

    def class_diversity_loss(self, batch_outputs):
        """
        计算改进的类多样性损失 - 增加对稀有类别的权重，添加数值稳定性保护
        """
        softmax_outputs = torch.softmax(batch_outputs, dim=1)

        # 计算每个类别的平均置信度
        alpha_j = softmax_outputs.mean(dim=0)  # shape: [num_classes]

        # 计算当前批次中各类别的出现次数
        _, predicted = torch.max(batch_outputs, dim=1)
        class_counts = torch.zeros(10, device=device)
        for i in range(10):
            class_counts[i] = (predicted == i).sum().item()

        # 对缺失或稀少的类别增加权重 - 但设置合理的上限
        class_weights = torch.ones(10, device=device)
        for i in range(10):
            if class_counts[i] < 1:
                # 缺失类别的权重增加，但不要过大
                class_weights[i] = 3.0
            elif class_counts[i] < batch_outputs.size(0) / 20:
                # 稀少类别的权重适当增加
                class_weights[i] = 2.0

        # 添加梯度裁剪和数值稳定性保护
        weighted_alpha = alpha_j * class_weights
        # 确保值在合理范围内
        weighted_alpha = torch.clamp(weighted_alpha, 1e-7, 1e3)
        weighted_alpha = weighted_alpha / weighted_alpha.sum()  # 重新归一化

        # 计算负熵，添加数值稳定性保护: Lclass_div = ∑(j=0 to K) [αj log αj]
        log_term = torch.log(weighted_alpha + 1e-7)
        log_term = torch.clamp(log_term, -20, 20)  # 限制对数值范围避免数值不稳定
        loss = torch.sum(weighted_alpha * log_term)

        # 最终添加额外的安全检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("警告: 检测到类多样性损失为NaN或Inf，重置为小常数")
            loss = torch.tensor(0.1, device=device)

        return loss  # 最小化负熵

    def distribution_matching_loss(self, fake_features, real_features):
        """
        计算分布匹配损失（MMD）

        参数:
            fake_features: 生成样本的特征
            real_features: 真实样本的特征

        返回:
            mmd_loss: MMD损失
        """
        return self.compute_mmd(fake_features, real_features)

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

    def train(self, num_queries=8000000, g_steps=1, c_steps=1, evaluate_every=100000):
        """
        训练克隆模型，实现DFMS-HL算法结合批量主动学习

        参数:
            num_queries: 最大查询预算
            g_steps: 每次训练生成器的步数
            c_steps: 每次训练克隆模型的步数
            evaluate_every: 多少次查询后评估一次模型
        """
        print("开始带主动学习的DFMS-HL训练...")

        # 重置查询计数器
        initial_queries = self.query_count
        real_label = 1
        fake_label = 0

        # 保存训练历史
        history = {
            'g_loss': [],
            'div_loss': [],
            'adv_loss': [],
            'mmd_loss': [],
            'd_loss': [],
            'c_loss': [],
            'query_count': [],
            'accuracy': []
        }

        # 可视化初始类别分布
        print("初始类别分布:")
        self.visualize_class_distribution()

        # 创建一个进度条
        pbar = tqdm(total=num_queries, desc="主动学习DFMS-HL训练")
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
            mmd_loss_sum = 0

            for _ in range(g_steps):
                # 使用主动学习策略选择批次
                fake_images, _ = self.select_active_batch()
                batch_size = fake_images.size(0)

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

                # 分布匹配损失 (MMD)
                if real_data is not None and real_data.size(0) == batch_size:
                    with torch.no_grad():
                        real_features = self.feature_extractor(real_data)
                        fake_features = self.feature_extractor(fake_images)

                    mmd_loss = self.distribution_matching_loss(fake_features, real_features)
                    mmd_loss_sum += mmd_loss.item()
                else:
                    mmd_loss = torch.tensor(0.0, device=device)

                # 总生成器损失
                g_loss = adv_loss + self.lambda_div * div_loss + mmd_loss
                g_loss_sum += g_loss.item()
                g_loss.backward()
                self.optimizer_G.step()

            # 2. 训练克隆模型（固定生成器和判别器）
            self.generator.eval()
            self.clone_model.train()

            c_loss_sum = 0
            for _ in range(c_steps):
                # 使用主动学习策略选择批次
                fake_images, labels = self.select_active_batch()

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
            mmd_loss_avg = mmd_loss_sum / g_steps
            c_loss_avg = c_loss_sum / c_steps

            history['g_loss'].append(g_loss_avg)
            history['div_loss'].append(div_loss_avg)
            history['adv_loss'].append(adv_loss_avg)
            history['mmd_loss'].append(mmd_loss_avg)
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

            if self.query_count - initial_queries >= 500000 and (self.query_count - initial_queries) % 500000 == 0:
                print("执行生成器部分重置以避免局部最优...")
                # 只重置生成器的部分层，保留一些学到的特征
                for name, param in self.generator.named_parameters():
                    # 只重置最后两层的参数，保留前面层的知识
                    if 'main.8' in name or 'main.10' in name:
                        if 'weight' in name:
                            nn.init.normal_(param.data, 0.0, 0.02)
                        elif 'bias' in name:
                            nn.init.constant_(param.data, 0)

                # 增加噪声池大小，促进更广泛的探索
                self.noise_pool_size = min(self.noise_pool_size * 2, 10000)
                self.noise_pool = torch.randn(self.noise_pool_size, self.nz, device=device)

                # 随机打乱迭代器以获取新的代理数据分布
                if hasattr(self, 'proxy_loader') and self.proxy_loader is not None:
                    self.proxy_iterator = iter(self.proxy_loader)

            # 检查是否超出查询预算
            if self.query_count - initial_queries >= num_queries:
                break

        pbar.close()

        # 保存最终模型和训练历史
        torch.save(self.clone_model.state_dict(), 'clone_model_final.pth')
        self._save_training_history(history)

        print(f"主动学习DFMS-HL训练完成，总查询数: {self.query_count}")
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

    # 初始化带主动学习的DFMS_HL
    # 选择主动学习策略：'uncertainty', 'boundary', 'cluster', 'combined'
    dfms_hl = DFMS_HL_Active(
        victim_model,
        generator,
        proxy_dataset=proxy_dataset,
        nc=nc,
        lambda_div=500,
        active_strategy='combined',  # 组合策略通常效果最好
        k=10,  # 聚类数量
        batch_size=128
    )

    # 初始化克隆模型
    # dfms_hl.initialize_clone(num_iterations=20000, evaluate_every=20000)

    # 主训练过程
    history = dfms_hl.train(
        num_queries=8000000,  # 论文中提到的800万查询预算
        g_steps=1,  # 每次迭代训练生成器1次
        c_steps=1,  # 每次迭代训练克隆模型1次，确保严格交替
        evaluate_every=100000  # 每10万次查询评估一次
    )