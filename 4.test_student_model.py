import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# 支持中文
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义数据预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# 加载CIFAR-10测试数据集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义模型函数
def create_resnet18_model():
    """创建ResNet-18模型"""
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def test_model(model, dataloader, model_name="Model"):
    """测试模型准确率"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 收集预测和目标用于混淆矩阵
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = 100. * correct / total
    print(f'{model_name}测试准确率: {acc:.2f}%')

    return acc, all_preds, all_targets


def plot_confusion_matrix(y_true, y_pred, classes, model_name="Model", suffix="",output_dir="."):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')

    filename = f'{model_name.lower().replace(" ", "_")}_confusion_matrix{suffix}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()


def compare_models(victim_model, clone_model, dataloader, suffix="",output_dir="."):
    """比较目标模型和克隆模型的性能"""
    # 测试目标模型
    victim_acc, victim_preds, victim_targets = test_model(victim_model, dataloader, "目标模型")

    # 测试克隆模型
    clone_acc, clone_preds, clone_targets = test_model(clone_model, dataloader, "克隆模型")

    # 计算克隆模型与目标模型的预测一致率
    agreement = np.mean(np.array(victim_preds) == np.array(clone_preds)) * 100
    print(f"克隆模型与目标模型的预测一致率: {agreement:.2f}%")

    # 绘制混淆矩阵
    plot_confusion_matrix(victim_targets, victim_preds, classes, "目标模型", suffix="_victim", output_dir=output_dir)
    plot_confusion_matrix(clone_targets, clone_preds, classes, "克隆模型", suffix=suffix, output_dir=output_dir)

    # 生成详细分类报告
    print("\n目标模型分类报告:")
    print(classification_report(victim_targets, victim_preds, target_names=classes))

    print("\n克隆模型分类报告:")
    print(classification_report(clone_targets, clone_preds, target_names=classes))

    # 绘制类别准确率比较
    victim_class_correct = np.zeros(10)
    victim_class_total = np.zeros(10)
    clone_class_correct = np.zeros(10)
    clone_class_total = np.zeros(10)

    for i in range(len(victim_targets)):
        victim_class_total[victim_targets[i]] += 1
        if victim_preds[i] == victim_targets[i]:
            victim_class_correct[victim_targets[i]] += 1

    for i in range(len(clone_targets)):
        clone_class_total[clone_targets[i]] += 1
        if clone_preds[i] == clone_targets[i]:
            clone_class_correct[clone_targets[i]] += 1

    victim_class_acc = victim_class_correct / victim_class_total * 100
    clone_class_acc = clone_class_correct / clone_class_total * 100

    plt.figure(figsize=(12, 6))
    x = np.arange(10)
    width = 0.35
    plt.bar(x - width / 2, victim_class_acc, width, label='目标模型')
    plt.bar(x + width / 2, clone_class_acc, width, label='克隆模型')
    plt.xlabel('类别')
    plt.ylabel('准确率 (%)')
    plt.title('各类别准确率比较')
    plt.xticks(x, classes)
    plt.legend()

    filename = f'class_accuracy_comparison{suffix}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()

    return victim_acc, clone_acc, agreement


# 主函数
if __name__ == "__main__":
    # --- 1. 定义所有文件路径 ---
    VICTIM_MODEL_PATH = "results/0_victim_model/victim_model.pth"
    BASELINE_CLONE_PATH = "results/2_baseline_attack/best_clone_model.pth"
    INNOVATION_CLONE_PATH = "results/3_innovation_attack/best_clone_model.pth"
    OUTPUT_DIR = "results/4_evaluation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. 在这里选择要测试的模型 ---
    # 要评估基线模型, 设置 experiment_type = 'baseline'
    # 要评估创新模型, 设置 experiment_type = 'innovation'
    experiment_type = 'innovation'  # <--- 在这里切换 'baseline' 或 'innovation'

    # --- 3. 【修正】根据选择, 确定正确的模型路径和输出文件后缀 ---
    if experiment_type == 'baseline':
        clone_model_path = BASELINE_CLONE_PATH
        output_suffix = '_baseline'
    else:  # innovation
        clone_model_path = INNOVATION_CLONE_PATH
        output_suffix = '_innovation'

    # --- 4. 加载模型 ---
    # 加载目标模型
    victim_model = create_resnet18_model()
    victim_model.load_state_dict(torch.load(VICTIM_MODEL_PATH))
    victim_model = victim_model.to(device)

    # 加载克隆模型
    print(f"正在加载克隆模型: {clone_model_path}") # <-- 使用修正后的 clone_model_path
    clone_model = create_resnet18_model()
    clone_model.load_state_dict(torch.load(clone_model_path))  # <-- 使用修正后的 clone_model_path
    clone_model = clone_model.to(device)

    # --- 5. 评估和比较两个模型 ---
    victim_acc, clone_acc, agreement = compare_models(victim_model, clone_model, testloader, output_suffix, OUTPUT_DIR)


    # --- 6. 打印并保存总结 ---
    print("\n=== 模型窃取性能总结 ===")
    print(f"实验类型: {experiment_type}")
    print(f"目标模型准确率: {victim_acc:.2f}%")
    print(f"克隆模型准确率: {clone_acc:.2f}%")
    print(f"克隆模型与目标模型的预测一致率: {agreement:.2f}%")
    print(f"性能比例 (克隆/目标): {clone_acc / victim_acc * 100:.2f}%")

    # 保存结果到文件
    summary_filename = f'results_summary{output_suffix}.txt'
    summary_path = os.path.join(OUTPUT_DIR, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 模型窃取性能总结 ({experiment_type}) ===\n")
        f.write(f"目标模型准确率: {victim_acc:.2f}%\n")
        f.write(f"克隆模型准确率: {clone_acc:.2f}%\n")
        f.write(f"克隆模型与目标模型的预测一致率: {agreement:.2f}%\n")
        f.write(f"性能比例 (克隆/目标): {clone_acc / victim_acc * 100:.2f}%\n")
    print(f"总结报告已保存至 {summary_path}")
