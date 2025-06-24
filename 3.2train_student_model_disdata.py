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
from PIL import Image
import copy

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
plt.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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


# 基于数据蒸馏的模型窃取实现 - 改进版
class EnhancedDatasetDistillation:
    def __init__(self, victim_model, proxy_dataset, num_synthetic=200, image_size=32, channels=3):
        """
        初始化增强版数据蒸馏模型窃取类

        参数:
            victim_model: 目标模型
            proxy_dataset: 代理数据集
            num_synthetic: 要生成的合成数据点数量
            image_size: 图像大小
            channels: 图像通道数
        """
        self.victim_model = victim_model
        self.victim_model.eval()  # 确保目标模型处于评估模式

        # 创建代理数据加载器
        self.proxy_dataset = proxy_dataset
        self.proxy_loader = DataLoader(proxy_dataset, batch_size=128, shuffle=True, num_workers=0)

        # 创建克隆模型
        self.clone_model = create_clone_model().to(device)

        # 初始化合成数据
        self.num_synthetic = num_synthetic
        self.image_size = image_size
        self.channels = channels

        # 为每个类别分配固定数量的合成数据点
        samples_per_class = num_synthetic // 10
        self.class_indices = {}
        for cls in range(10):
            start_idx = cls * samples_per_class
            end_idx = start_idx + samples_per_class if cls < 9 else num_synthetic
            self.class_indices[cls] = list(range(start_idx, end_idx))

        # 随机初始化合成数据
        self.synthetic_data = nn.Parameter(
            torch.randn(num_synthetic, channels, image_size, image_size, device=device) * 0.1
        )

        # 配置类别标签（固定分配）
        self.synthetic_labels = torch.zeros(num_synthetic, dtype=torch.long, device=device)
        for cls, indices in self.class_indices.items():
            self.synthetic_labels[indices] = cls

        # 存储软标签（概率分布）
        self.soft_labels = torch.zeros(num_synthetic, 10, device=device)

        # 查询计数器
        self.query_count = 0

        # 损失函数
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.criterion_mse = nn.MSELoss()

        # 创建合成数据优化器和学习率调度器
        self.optimizer_data = optim.Adam([self.synthetic_data], lr=0.05)
        self.scheduler_data = optim.lr_scheduler.StepLR(self.optimizer_data, step_size=300, gamma=0.5)

        # 克隆模型的优化器
        self.optimizer_clone = optim.SGD(self.clone_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler_clone = CosineAnnealingLR(self.optimizer_clone, T_max=200)

        # 最佳模型跟踪
        self.best_clone_acc = 0.0
        self.best_clone_state = None

        # 创建测试数据加载器用于验证
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=0)

        # 数据增强转换
        self.augmentation = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def _get_soft_label(self, x, temperature=3.0):
        """从目标模型获取软标签（概率分布）"""
        with torch.no_grad():
            outputs = self.victim_model(x)
            soft_targets = torch.softmax(outputs / temperature, dim=1)
            self.query_count += x.size(0)
        return soft_targets

    def _get_hard_label(self, x):
        """从目标模型获取硬标签"""
        with torch.no_grad():
            outputs = self.victim_model(x)
            _, predicted = outputs.max(1)
            self.query_count += x.size(0)
        return predicted

    def _normalize_synthetic_data(self):
        """标准化合成数据，确保在合理范围内"""
        with torch.no_grad():
            self.synthetic_data.data.clamp_(-1, 1)

    def _apply_augmentation(self, images):
        """应用数据增强到张量图像"""
        # 转换为PIL图像，应用变换，再转回张量
        batch_size = images.size(0)
        augmented = torch.zeros_like(images)

        # 归一化到[0,1]范围以使用PIL
        normalized = (images + 1) / 2

        for i in range(batch_size):
            img = transforms.ToPILImage()(normalized[i])
            aug_img = self.augmentation(img)
            aug_tensor = transforms.ToTensor()(aug_img)
            augmented[i] = aug_tensor * 2 - 1  # 转回[-1,1]范围

        return augmented

    def _compute_feature_matching_loss(self, student_model, teacher_model, x, layer_weights=None):
        """计算特征匹配损失，带层权重"""
        if layer_weights is None:
            layer_weights = [1.0, 1.0, 2.0, 3.0]  # 更深的层权重更大

        # 提取多层特征
        teacher_features = self._extract_multiple_features(teacher_model, x, detach=True)
        student_features = self._extract_multiple_features(student_model, x, detach=False)

        # 计算加权特征匹配损失
        loss = 0
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            if i < len(layer_weights):
                weight = layer_weights[i]
                loss += weight * self.criterion_mse(s_feat, t_feat)

        return loss

    def _extract_multiple_features(self, model, x, detach=True):
        """提取模型多个层的特征"""
        features = []
        activations = {}

        # 定义钩子函数
        def get_activation(name):
            def hook(model, input, output):
                if detach:
                    activations[name] = output.detach()
                else:
                    activations[name] = output

            return hook

        # 注册钩子
        handles = []
        handles.append(model.layer1.register_forward_hook(get_activation('layer1')))
        handles.append(model.layer2.register_forward_hook(get_activation('layer2')))
        handles.append(model.layer3.register_forward_hook(get_activation('layer3')))
        handles.append(model.layer4.register_forward_hook(get_activation('layer4')))

        # 前向传播
        model(x)

        # 收集特征
        features = [
            activations['layer1'].mean(dim=[2, 3]),
            activations['layer2'].mean(dim=[2, 3]),
            activations['layer3'].mean(dim=[2, 3]),
            activations['layer4'].mean(dim=[2, 3])
        ]

        # 移除钩子
        for handle in handles:
            handle.remove()

        return features

    def _compute_boundary_seeking_loss(self, x, model):
        """计算决策边界寻找损失 - 鼓励生成靠近决策边界的样本"""
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

        # 获取前两个最高概率
        top_probs, _ = torch.topk(probs, k=2, dim=1)
        p1, p2 = top_probs[:, 0], top_probs[:, 1]

        # 鼓励这两个概率接近，表示样本靠近决策边界
        boundary_loss = -torch.mean(1 - (p1 - p2))

        return boundary_loss

    def _compute_consistency_loss(self, x, y_soft):
        """计算一致性损失 - 确保克隆模型输出与目标模型接近"""
        # 前向传播获取克隆模型输出
        outputs = self.clone_model(x)
        probs = torch.log_softmax(outputs, dim=1)

        # 计算KL散度
        consistency_loss = self.criterion_kl(probs, y_soft)

        return consistency_loss

    def _initialize_synthetic_data(self):
        """从代理数据集初始化合成数据"""
        print("初始化合成数据...")

        # 从代理数据中收集每个类别的样本
        class_samples = {cls: [] for cls in range(10)}

        # 查询目标模型获取代理数据的类别
        with torch.no_grad():
            for x, _ in self.proxy_loader:
                x = x.to(device)
                batch_size = x.size(0)
                outputs = self.victim_model(x)
                _, pred_classes = outputs.max(1)
                soft_labels = torch.softmax(outputs, dim=1)
                self.query_count += batch_size

                # 收集每个类别的样本和对应的软标签
                for i in range(batch_size):
                    cls = pred_classes[i].item()
                    if cls in class_samples and len(class_samples[cls]) < len(self.class_indices[cls]):
                        class_samples[cls].append((x[i].detach().clone(), soft_labels[i].detach().clone()))

                # 检查是否所有类别都已收集足够样本
                if all(len(samples) >= len(self.class_indices[cls]) for cls, samples in class_samples.items() if
                       cls in self.class_indices):
                    break

        # 使用收集的样本初始化合成数据
        for cls, indices in self.class_indices.items():
            if cls in class_samples and class_samples[cls]:
                for idx, synth_idx in enumerate(indices):
                    if idx < len(class_samples[cls]):
                        sample, soft_label = class_samples[cls][idx]
                        # 添加小噪声以增加多样性
                        self.synthetic_data.data[synth_idx] = sample + 0.05 * torch.randn_like(sample)
                        self.soft_labels[synth_idx] = soft_label

        print("合成数据初始化完成")

    def optimize_synthetic_data(self, num_iterations=1000, alternate_every=10):
        """
        优化合成数据，使用交替训练策略

        参数:
            num_iterations: 总迭代次数
            alternate_every: 每多少次迭代交替训练一次克隆模型
        """
        print("开始优化合成数据...")

        # 初始化合成数据
        self._initialize_synthetic_data()

        # 存储训练历史
        history = {
            'feature_loss': [],
            'boundary_loss': [],
            'consistency_loss': [],
            'clone_loss': [],
            'total_loss': [],
            'clone_acc': [],
            'query_count': []
        }

        # 记录初始分布情况
        print("初始标签分布:")
        self._print_label_distribution()

        # 设置进度条
        pbar = tqdm(total=num_iterations, desc="优化合成数据")

        # 设置温度参数，用于软标签蒸馏
        temperature = 3.0

        for iteration in range(num_iterations):
            # 阶段1: 优化合成数据
            if iteration % alternate_every == 0:
                # 每alternate_every次迭代，切换到训练克隆模型
                self._train_clone_model_epoch(temperature)

            # 阶段2: 更新合成数据
            self.optimizer_data.zero_grad()

            # 1. 计算特征匹配损失
            # 随机选择代理数据
            for real_x, _ in self.proxy_loader:
                real_x = real_x.to(device)
                feature_loss = self._compute_feature_matching_loss(
                    self.clone_model, self.victim_model, real_x
                )
                break

            # 2. 计算决策边界寻找损失
            # 随机选择一批合成数据
            indices = []
            for cls in range(10):
                if cls in self.class_indices:
                    # 从每个类别选择几个样本
                    cls_indices = random.sample(self.class_indices[cls], min(5, len(self.class_indices[cls])))
                    indices.extend(cls_indices)

            if indices:
                boundary_batch = self.synthetic_data[indices]
                boundary_loss = self._compute_boundary_seeking_loss(boundary_batch, self.victim_model)
            else:
                boundary_loss = torch.tensor(0.0, device=device)

            # 3. 计算一致性损失
            # 使用所有合成数据
            with torch.no_grad():
                soft_targets = self._get_soft_label(self.synthetic_data, temperature)

            # 对合成数据应用数据增强
            augmented_data = self._apply_augmentation(self.synthetic_data)
            consistency_loss = self._compute_consistency_loss(augmented_data, soft_targets)

            # 总损失 - 加权组合
            total_loss = feature_loss + 0.5 * boundary_loss + 2.0 * consistency_loss

            # 反向传播和优化
            total_loss.backward()
            self.optimizer_data.step()
            self.scheduler_data.step()

            # 标准化合成数据
            self._normalize_synthetic_data()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(
                f_loss=f"{feature_loss.item():.4f}",
                b_loss=f"{boundary_loss.item():.4f}",
                c_loss=f"{consistency_loss.item():.4f}",
                queries=self.query_count
            )

            # 每100次迭代更新软标签
            if iteration % 100 == 0:
                with torch.no_grad():
                    self.soft_labels = self._get_soft_label(self.synthetic_data, temperature)
                    # 同时更新硬标签（仅用于评估）
                    hard_labels = self._get_hard_label(self.synthetic_data)
                    for cls, indices in self.class_indices.items():
                        for idx in indices:
                            self.synthetic_labels[idx] = hard_labels[idx]

            # 每200次迭代评估性能
            if iteration % 200 == 0:
                # 完整训练克隆模型
                self.train_clone_model(num_epochs=30, batch_size=64, temperature=temperature)
                acc = self.evaluate_clone()

                # 记录训练历史
                history['feature_loss'].append(feature_loss.item())
                history['boundary_loss'].append(boundary_loss.item())
                history['consistency_loss'].append(consistency_loss.item())
                history['total_loss'].append(total_loss.item())
                history['clone_acc'].append(acc)
                history['query_count'].append(self.query_count)

                # 打印标签分布
                print(f"\n迭代 {iteration}, 标签分布:")
                self._print_label_distribution()

                # 保存中间结果的合成数据样本
                self.visualize_synthetic_data(num_samples=10, save_path=f'synthetic_samples_iter_{iteration}.png')

        pbar.close()
        print("合成数据优化完成")

        # 最终标签分布
        print("最终标签分布:")
        self._print_label_distribution()

        # 再次使用最终优化的合成数据全面训练克隆模型
        print("使用最终优化的合成数据训练克隆模型...")
        self.train_clone_model(num_epochs=100, batch_size=64, temperature=temperature)
        final_acc = self.evaluate_clone()

        # 绘制训练历史并保存合成数据
        self._plot_history(history)
        self._save_synthetic_dataset()

        # 恢复最佳模型状态
        if self.best_clone_state is not None:
            self.clone_model.load_state_dict(self.best_clone_state)
            print(f"恢复最佳克隆模型，准确率: {self.best_clone_acc:.2f}%")

        return history, final_acc

    def _train_clone_model_epoch(self, temperature=1.0):
        """训练克隆模型一个epoch"""
        self.clone_model.train()

        # 创建合成数据的数据加载器
        indices = list(range(self.num_synthetic))
        random.shuffle(indices)
        batch_size = min(64, len(indices))

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            inputs = self.synthetic_data[batch_indices]
            hard_targets = self.synthetic_labels[batch_indices]
            soft_targets = self.soft_labels[batch_indices]

            # 数据增强
            inputs = self._apply_augmentation(inputs)

            self.optimizer_clone.zero_grad()

            # 前向传播
            outputs = self.clone_model(inputs)

            # 硬标签损失（交叉熵）
            hard_loss = self.criterion_ce(outputs, hard_targets)

            # 软标签损失（KL散度）
            log_probs = torch.log_softmax(outputs / temperature, dim=1)
            soft_loss = self.criterion_kl(log_probs, soft_targets) * (temperature ** 2)

            # 总损失 - 加权组合
            loss = 0.5 * hard_loss + 0.5 * soft_loss

            # 反向传播和优化
            loss.backward()
            self.optimizer_clone.step()

        self.scheduler_clone.step()

    def train_clone_model(self, num_epochs=50, batch_size=32, temperature=1.0):
        """使用优化后的合成数据完整训练克隆模型"""
        print(f"训练克隆模型 (epochs={num_epochs})...")

        # 重置克隆模型
        self.clone_model = create_clone_model().to(device)
        self.optimizer_clone = optim.SGD(self.clone_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler_clone = CosineAnnealingLR(self.optimizer_clone, T_max=num_epochs)

        # 创建合成数据的数据加载器
        synthetic_dataset = TensorDataset(self.synthetic_data, self.synthetic_labels)
        synthetic_loader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)

        # 训练克隆模型
        self.clone_model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0

            for inputs, labels in synthetic_loader:
                # 数据增强
                inputs = self._apply_augmentation(inputs)

                self.optimizer_clone.zero_grad()

                # 前向传播
                outputs = self.clone_model(inputs)

                # 从软标签中提取当前批次的软标签
                batch_soft_labels = torch.stack([self.soft_labels[i] for i in range(len(labels))])

                # 硬标签损失
                hard_loss = self.criterion_ce(outputs, labels)

                # 软标签损失
                log_probs = torch.log_softmax(outputs / temperature, dim=1)
                soft_loss = self.criterion_kl(log_probs, batch_soft_labels) * (temperature ** 2)

                # 总损失 - 加权组合
                loss = 0.5 * hard_loss + 0.5 * soft_loss

                # 反向传播和优化
                loss.backward()
                self.optimizer_clone.step()

                running_loss += loss.item()

            # 更新学习率
            self.scheduler_clone.step()

            # 每10个epoch评估一次
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(synthetic_loader):.4f}")
                acc = self.evaluate_clone()

                # 如果是最佳模型，保存状态
                if acc > self.best_clone_acc:
                    self.best_clone_acc = acc
                    self.best_clone_state = copy.deepcopy(self.clone_model.state_dict())
                    torch.save(self.clone_model.state_dict(), 'best_clone_model.pth')
                    print(f"保存新的最佳模型，准确率: {acc:.2f}%")

        print("克隆模型训练完成")

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

        return accuracy

    def _print_label_distribution(self):
        """打印合成数据标签分布"""
        counts = torch.zeros(10, dtype=torch.int)
        for label in self.synthetic_labels:
            counts[label] += 1

        percentage = counts.float() / counts.sum() * 100

        for i in range(10):
            print(f"类别 {i}: {counts[i].item()} ({percentage[i].item():.2f}%)")

    def _plot_label_distribution(self, save_path='label_distribution.png'):
        """绘制合成数据的标签分布"""
        counts = torch.zeros(10, dtype=torch.int)
        for label in self.synthetic_labels:
            counts[label] += 1

        percentage = counts.float() / counts.sum() * 100

        plt.figure(figsize=(10, 6))
        plt.bar(range(10), percentage.cpu().numpy())
        plt.xlabel('类别')
        plt.ylabel('百分比 (%)')
        plt.title('合成数据的类别分布')
        plt.xticks(range(10))
        plt.savefig(save_path)
        plt.close()

    def _plot_history(self, history):
        """绘制训练历史"""
        plt.figure(figsize=(15, 15))

        # 绘制损失
        plt.subplot(3, 1, 1)
        plt.plot(history['feature_loss'], label='特征匹配损失')
        plt.plot(history['boundary_loss'], label='决策边界损失')
        plt.plot(history['consistency_loss'], label='一致性损失')
        plt.plot(history['total_loss'], label='总损失')
        plt.title('优化损失')
        plt.xlabel('评估次数')
        plt.ylabel('损失')
        plt.legend()

        # 绘制准确率
        plt.subplot(3, 1, 2)
        plt.plot(history['clone_acc'], label='克隆模型准确率')
        plt.title('克隆模型准确率')
        plt.xlabel('评估次数')
        plt.ylabel('准确率 (%)')
        plt.legend()

        # 绘制查询次数
        plt.subplot(3, 1, 3)
        plt.plot(history['query_count'], label='查询次数')
        plt.title('目标模型查询次数')
        plt.xlabel('评估次数')
        plt.ylabel('查询次数')
        plt.legend()

        plt.tight_layout()
        plt.savefig('enhanced_distillation_history.png')
        plt.close()

    def _save_synthetic_dataset(self, filename='enhanced_synthetic_dataset.pt'):
        """保存合成数据集"""
        torch.save({
            'data': self.synthetic_data.detach().cpu(),
            'hard_labels': self.synthetic_labels.cpu(),
            'soft_labels': self.soft_labels.cpu()
        }, filename)
        print(f"合成数据集已保存到 {filename}")

    def visualize_synthetic_data(self, num_samples=10, save_path='enhanced_synthetic_samples.png'):
        """可视化合成数据样本"""
        # 选择每个类别的一个样本（如果有）
        samples_to_show = []
        labels_to_show = []

        for cls in range(10):
            if cls in self.class_indices and self.class_indices[cls]:
                for idx in self.class_indices[cls]:
                    samples_to_show.append(self.synthetic_data[idx])
                    labels_to_show.append(cls)
                    break

        # 如果样本数不足，添加更多样本
        if len(samples_to_show) < num_samples:
            additional_indices = random.sample(range(self.num_synthetic), num_samples - len(samples_to_show))
            samples_to_show.extend([self.synthetic_data[i] for i in additional_indices])
            labels_to_show.extend([self.synthetic_labels[i].item() for i in additional_indices])

        # 截断到指定数量
        samples_to_show = samples_to_show[:num_samples]
        labels_to_show = labels_to_show[:num_samples]

        # 反归一化图像
        denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))

        plt.figure(figsize=(12, 6))
        for i, (img, label) in enumerate(zip(samples_to_show, labels_to_show)):
            plt.subplot(2, 5, i + 1)
            img_np = denorm(img).detach().cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)

            plt.imshow(img_np)
            plt.title(f'类别: {label}')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"合成数据样本已保存到 {save_path}")


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

    # 使用CIFAR-100代理数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 选择40个类别的CIFAR-100作为代理数据
    proxy_dataset = get_proxy_dataset(num_classes=40)

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

    # 初始化增强版数据蒸馏模型窃取
    model_thief = EnhancedDatasetDistillation(
        victim_model=victim_model,
        proxy_dataset=proxy_dataset,
        num_synthetic=200,  # 生成200个合成数据点
        image_size=32,
        channels=3
    )

    # 执行优化过程
    history, final_acc = model_thief.optimize_synthetic_data(num_iterations=2000, alternate_every=10)

    # 可视化最终合成数据
    model_thief.visualize_synthetic_data(num_samples=10)
    model_thief._plot_label_distribution()

    # 输出最终结果
    print(f"最终克隆模型准确率: {final_acc:.2f}%")
    print(f"最佳克隆模型准确率: {model_thief.best_clone_acc:.2f}%")
    print(f"总查询次数: {model_thief.query_count}")