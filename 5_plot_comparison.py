import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 配置 ---
# 定义包含训练历史记录的 .pkl 文件的路径
BASELINE_HISTORY_PATH = "results/2_baseline_attack/training_history.pkl"
INNOVATION_HISTORY_PATH = "results/3_innovation_attack/training_history.pkl"

# 定义图表保存的路径和文件名
OUTPUT_DIR = "results/4_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURE_SAVE_PATH = os.path.join(OUTPUT_DIR, "accuracy_vs_queries_comparison.png")

# 图表标题和标签（支持中文）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
CHART_TITLE = '基线攻击 vs 创新攻击 性能对比'
X_LABEL = '查询次数 (Queries)'
Y_LABEL = '克隆模型准确率 (%)'
BASELINE_LABEL = '基线攻击 (Baseline Attack)'
INNOVATION_LABEL = '创新攻击 (Innovation Attack)'

# --- 主程序 ---

def plot_comparison(baseline_path, innovation_path, save_path):
    """
    加载两个模型的训练历史，并绘制准确率与查询次数的关系图。
    """
    try:
        # 加载基线模型的历史数据
        with open(baseline_path, 'rb') as f:
            baseline_history = pickle.load(f)
        print(f"成功加载基线模型历史数据: {baseline_path}")

        # 加载创新模型的历史数据
        with open(innovation_path, 'rb') as f:
            innovation_history = pickle.load(f)
        print(f"成功加载创新模型历史数据: {innovation_path}")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。请确保您已经成功运行了训练脚本并生成了 .pkl 文件。")
        return
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return

    # --- 数据处理与绘图 ---
    plt.figure(figsize=(12, 8)) # 创建一个较大的画布

    # --- 绘制基线模型数据 ---
    # 论文中的evaluate_every是100000，训练代码中也是如此
    # 我们需要为准确率数据点生成对应的查询次数x轴坐标
    # 加上初始化查询数 20000*128=2560000，才是总查询数
    initial_queries_baseline = baseline_history['query_count'][0] if 'query_count' in baseline_history and baseline_history['query_count'] else 0
    num_evaluations_baseline = len(baseline_history['accuracy'])
    evaluate_every_baseline = (8000000 - 2560000) / num_evaluations_baseline # 动态计算间隔
    
    # x_baseline = [initial_queries_baseline + (i + 1) * evaluate_every_baseline for i in range(num_evaluations_baseline)]
    x_baseline = np.linspace(2560000, 8000000, num=num_evaluations_baseline)
    y_baseline = baseline_history['accuracy']
    
    plt.plot(x_baseline, y_baseline, marker='o', linestyle='-', color='royalblue', label=BASELINE_LABEL)

    # --- 绘制创新模型数据 ---
    initial_queries_innovation = innovation_history['query_count'][0] if 'query_count' in innovation_history and innovation_history['query_count'] else 0
    num_evaluations_innovation = len(innovation_history['accuracy'])
    evaluate_every_innovation = (8000000 - 2560000) / num_evaluations_innovation # 动态计算间隔

    # x_innovation = [initial_queries_innovation + (i + 1) * evaluate_every_innovation for i in range(num_evaluations_innovation)]
    x_innovation = np.linspace(2560000, 8000000, num=num_evaluations_innovation)
    y_innovation = innovation_history['accuracy']
    
    plt.plot(x_innovation, y_innovation, marker='s', linestyle='-', color='crimson', label=INNOVATION_LABEL)


    # --- 图表美化 ---
    plt.title(CHART_TITLE, fontsize=16)
    plt.xlabel(X_LABEL, fontsize=12)
    plt.ylabel(Y_LABEL, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6) # 添加网格线
    plt.legend(fontsize=12) # 显示图例

    # 格式化x轴，使其更易读（例如，显示为 "2M", "4M"）
    def millions_formatter(x, pos):
        return f'{x / 1e6:.1f}M'
    
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(millions_formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tight_layout() # 调整布局以防止标签重叠

    # --- 保存图表 ---
    plt.savefig(save_path, dpi=300) # 保存为高分辨率图像
    print(f"对比图已成功保存至: {save_path}")
    
    # 显示图表（如果您在本地运行并希望立即看到它）
    # plt.show()


if __name__ == "__main__":
    plot_comparison(BASELINE_HISTORY_PATH, INNOVATION_HISTORY_PATH, FIGURE_SAVE_PATH)
