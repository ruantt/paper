# 6_visualize_feature_space.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from tqdm import tqdm

# --- 0. 环境与路径配置 ---

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义模型路径
VICTIM_MODEL_PATH = "results/0_victim_model/victim_model.pth"
BASELINE_CLONE_PATH = "results/2_baseline_attack/best_clone_model.pth"
INNOVATION_CLONE_PATH = "results/3_innovation_attack/best_clone_model.pth"

# 定义结果保存路径
OUTPUT_DIR = "results/5_feature_space_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 模型定义与加载 ---

# 定义与您训练时完全相同的模型结构
def create_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# 加载三个模型
print("正在加载模型...")
# 受害者模型 V
victim_model = create_model().to(device)
victim_model.load_state_dict(torch.load(VICTIM_MODEL_PATH, map_location=device))
victim_model.eval()

# 基线克隆模型 C_base
baseline_clone_model = create_model().to(device)
baseline_clone_model.load_state_dict(torch.load(BASELINE_CLONE_PATH, map_location=device))
baseline_clone_model.eval()

# 创新克隆模型 C_innovation
innovation_clone_model = create_model().to(device)
innovation_clone_model.load_state_dict(torch.load(INNOVATION_CLONE_PATH, map_location=device))
innovation_clone_model.eval()

print("所有模型加载完毕。")

# --- 2. 数据加载 ---

# 加载标准的CIFAR-10测试集
# 注意：这里的 transform 必须和训练受害者模型时用的测试集 transform 完全一致
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# 使用较大的 batch_size 以加速特征提取
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# CIFAR-10 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# --- 3. 特征提取核心逻辑 ---

features_map = {}
def get_features(name):
    """一个钩子函数，用于捕获指定层的输出"""
    def hook(model, input, output):
        # 我们需要的是进入全连接层之前的特征，所以是 output.detach()
        features_map[name] = output.detach()
    return hook

def extract_features(model, dataloader, hook_handle):
    """提取整个数据集的特征和标签"""
    all_features = []
    all_labels = []

    with torch.no_grad():
        # 使用tqdm显示进度条
        for images, labels in tqdm(dataloader, desc=f"提取 {model.name} 的特征"):
            images = images.to(device)
            # 模型进行一次前向传播，钩子函数会自动捕获特征
            model(images)
            # 从 features_map 中获取捕获到的特征
            batch_features = features_map['features']
            # 将特征从GPU移到CPU，并转为Numpy数组
            all_features.append(batch_features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 移除钩子，防止内存泄漏
    hook_handle.remove()
    
    # 将列表合并成大的Numpy数组
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

# --- 4. 依次为每个模型提取特征 ---

# 为每个模型命名，用于显示进度和保存文件
victim_model.name = "受害者模型 (Victim Model)"
baseline_clone_model.name = "基线克隆模型 (Baseline Clone)"
innovation_clone_model.name = "创新克隆模型 (Innovation Clone)"

models_to_visualize = [victim_model, baseline_clone_model, innovation_clone_model]
all_model_features = {}

for model in models_to_visualize:
    # 注册钩子到模型的倒数第二层（全连接层之前）
    # ResNet-18 的结构是 ... -> avgpool -> fc。我们需要 avgpool 的输出
    # 我们可以通过 model.avgpool 访问到它
    hook = model.avgpool.register_forward_hook(get_features('features'))
    
    # 执行特征提取
    features, labels = extract_features(model, testloader, hook)
    
    # 将提取到的特征和标签存储起来
    all_model_features[model.name] = (features.reshape(features.shape[0], -1), labels) # Reshape features to 2D


# --- 5. t-SNE 降维与可视化 ---

print("\n开始 t-SNE 降维与可视化 (这可能需要几分钟)...")

# 创建一个 1x3 的子图布局来并排显示三张图
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('不同模型在CIFAR-10测试集上的特征空间可视化', fontsize=20)

for i, model_name in enumerate(all_model_features.keys()):
    print(f"正在处理: {model_name}")
    features, labels = all_model_features[model_name]
    
    # 初始化t-SNE模型
    # n_components=2: 降到2维
    # perplexity: 通常在5-50之间，它关系到近邻点的数量
    # n_iter: 迭代次数
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=1)
    
    # 执行降维
    tsne_results = tsne.fit_transform(features)
    
    # 开始绘图
    ax = axes[i]
    # 使用 matplotlib 的 scatter 函数绘制散点图
    # c=labels: 让点的颜色由它们的真实标签决定
    # cmap='tab10': 使用一个有10种清晰颜色的颜色映射表
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    
    ax.set_title(model_name, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

# 创建一个共享的图例
legend1 = fig.legend(handles=scatter.legend_elements()[0], labels=classes, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))
fig.add_artist(legend1)

# 调整布局并保存图像
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以容纳标题和图例
save_path = os.path.join(OUTPUT_DIR, "feature_space_comparison.png")
plt.savefig(save_path, dpi=300)

print(f"\n可视化对比图已成功保存至: {save_path}")
