# 这是修改后的完整代码
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
import pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] # 设置字体为黑体
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
def load_victim_model(model_path):
    model = create_clone_model()
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(f"成功加载目标模型权重从 {model_path}")
    except Exception as e:
        print(f"加载目标模型时出错: {e}")
        return None
    model = model.to(device)
    model.eval()
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

# 定义判别器
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


# 创新算法主要实现
class DFMS_HL_Swift_AWUS(nn.Module):
    def __init__(self, victim_model, generator, proxy_dataset=None, nc=3, nz=100,
                 lambda_div=500, lambda_contrast=1.0, temperature=0.07,
                 feature_dim=128, queue_size=4096, alpha_decay=0.9,output_dir="results/3_innovation_attack"):
        """
        初始化增强版DFMS-HL算法，整合SwiftThief的对比学习和AWUS采样策略

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
            alpha_decay: AWUS采样的α衰减率
        """
        # 继承nn.Module的初始化
        super(DFMS_HL_Swift_AWUS, self).__init__()

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
        # 简单来说，它的作用可以概括为：充当一个“翻译官”和“思想学习者”，它负责将图片翻译成一种“思想”或“摘要”（即特征向量），并通过对比学习来模仿受害者模型“思考”的方式，而不仅仅是模仿它的最终“答案”。
        #
        # 下面我们来深入解析它的具体作用和工作流程：
        #
        # 1. 为什么需要一个独立的 feature_extractor？
        # 在基线方法中，克隆模型 C 的学习目标是：给定一张生成图片 x，我的输出要和受害者模型 V 的输出 ŷ(x) 一样。这是一个非常表层的模仿。
        #
        # 这就好比你学一个老师解题，你只关心老师最后的答案是不是"A"，而不去理解他推导出"A"的整个思路和过程。这样你可能在类似的题目上能答对，但遇到变种题就很容易出错。
        #
        # feature_extractor 的引入就是为了解决这个问题。它的目标是学习受害者模型的“解题思路”，即内部的特征表示。
        self.feature_extractor = self._create_feature_extractor().to(device)

        # 设置超参数
        self.nz = nz
        self.nc = nc
        self.lambda_div = lambda_div
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature
        self.alpha_decay = alpha_decay

        # 设置损失函数
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()

        # 查询计数器
        self.query_count = 0

        # 设置优化器
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_C = optim.SGD(self.clone_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # 特征提取器,它指导特征提取器F学习如何将图片转换成高质量的特征向量。它的唯一训练目标来自于对比损失 (contrast_loss)，即让同类样本的特征更近，异类样本的特征更远。
        self.optimizer_F = optim.SGD(self.feature_extractor.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # 学习率调度器,它会根据训练的进程，动态地调整学习率 lr。
        self.scheduler_C = CosineAnnealingLR(self.optimizer_C, T_max=200)
        self.scheduler_F = CosineAnnealingLR(self.optimizer_F, T_max=200)

        # 最佳模型跟踪
        self.best_clone_acc = 0.0

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
        #这部分代码的灵感来源于一个著名的对比学习算法叫做MoCo (Momentum Contrast)。它的核心思想是：为了让模型学会区分不同的样本，我们需要拿当前样本的特征去和大量的其他样本（负样本）的特征做对比。但如果一次性把成千上万的负样本都放进GPU，显存会爆炸。怎么办呢？答案就是创建一个“队列”。
        #作用: 创建一个用于存放特征向量的“队列”或“仓库”。
        self.register_buffer("queue", torch.randn(queue_size, feature_dim))
        #作用: 对队列里所有的特征向量进行“归一化”处理。
        self.queue = nn.functional.normalize(self.queue, dim=1)
        #作用: 创建一个“指针”，用来记录我们下一次应该在队列的哪个位置插入新的特征。
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        #作用: 把队列的大小保存为一个普通的成员变量，方便后续代码直接使用。
        self.queue_size = queue_size

        # 类别查询计数，用于优先级采样
        self.register_buffer("class_query_counts", torch.zeros(10))

        # AWUS参数初始化
        #讲解下AWUS的策略：
        # 第一步：感知学习状态(在evaluate_clone函数中self.previous_predictions: 记住上一次评估时，模型对所有测试图片的预测结果。计算self.model_change: 在本次评估时，用新的预测结果和上次的旧结果进行对比。model_change就是预测结果发生变化的图片所占的比例。
        # 如果模型很不稳定，每次评估结果都大变样，model_change的值就会很高（比如接近0.8）。
        # 如果模型已经很稳定，预测结果几乎不变，model_change的值就会很低（比如接近0.1）。
        #
        # 目的: 通过model_change这个指标，算法就能感知到克隆模型当前是处于“剧烈学习期”还是“稳定收敛期”。
        #
        # 第二步：决策采样策略(在_update_alpha函数中)
        #
        # 更新self.alpha: 算法会根据刚刚计算出的model_change来动态调整混合系数alpha。
        #
        # 如果model_change很高（不稳定），alpha也会被调高。
        #
        # 如果model_change很低（稳定），alpha也会被调低。
        #
        # 目的: alpha就是最终的决策旋钮。它决定了下一步行动中，“探索”和“挖掘”的权重。
        #
        # 第三步：执行智能行动(在generate_samples_with_awus函数中)这是最终执行采样的地方。
        #
        # 计算不确定性: 生成一批候选图片，然后用克隆模型预测它们，并计算每个图片的预测熵(entropy)。熵越高，代表模型对这个图片的分类越不确定。
        #
        # 混合权重: 使用上一步决策好的alpha值，将均匀权重（代表随机采样 / 探索）和不确定性权重（代表挖掘）进行加权混合：最终权重 = alpha * 随机权重 + (1 - alpha) * 不确定性权重
        #
        # 择优采样: 根据这个混合后的“最终权重”，从候选图片中进行抽样。
        #
        # 如果alpha很高，采样结果就更接近于随机，实现了“探索”。
        #
        # 如果alpha很低，采样结果就更倾向于选择那些熵很高的图片，实现了“挖掘”。
        #
        # 总结一下这个流程就是：
        #
        # 模型评估 ➔ 计算模型变化有多大(model_change) ➔ 决定该探索还是该挖掘(alpha) ➔ 执行加权采样


        # 1.
        # alphavs.model_change: 为什么要多一个
        # alpha？
        # “既然alpha和modelchange是同步变化为什么不用modelchange直接代替alpha？”这是一个非常好的问题！答案是：为了稳定性和平滑性。
        #
        # model_change：是一个瞬时指标。它只反映了“这次评估”和“上次评估”之间的变化，这个值可能会因为某一次训练的偶然性而剧烈波动。比如，模型可能在一个阶段稳定了一会儿（model_change很低），然后下个阶段又开始剧烈变化（model_change很高）。如果直接用
        # model_change来指导采样，我们的采样策略就会像“过山车”一样，忽左忽右，非常不稳定。
        # 而alpha是model_change的一个平滑版本。请看这行更新alpha的代码：
        # self.alpha = self.alpha * self.alpha_decay + target_alpha * (1 - self.alpha_decay)
        # 这是一种叫做“指数移动平均(Exponential Moving Average)”的技术。您可以把它理解为：
        # “新的alpha = 90 % 的旧alpha + 10 % 的当前model_change”。（假设alpha_decay是0.9）
        # 这样做的好处是：
        # 过滤噪声: alpha的值不会因为单次model_change的剧烈波动而大起大落。它考虑了历史的趋势。
        # 策略惯性: 使得采样策略的转变更加平滑、有惯性，避免了在“探索”和“挖掘”之间过于频繁和剧烈的切换，让整个学习过程更稳定。
        # 所以，model_change是原始的、嘈杂的信号，而alpha是经过平滑处理的、更可靠的决策依据。

        #2.2. “熵”与“不确定性”：此“熵”非彼“熵”
        # “计算不确定性怎么和熵联系起来的？为什么熵越大表示模型对图片分类越不稳定？熵不是衡量样本类别的吗？”
        #
        # 您的问题点出了一个关键概念：这里我们用的是“预测熵 (Predictive Entropy)”，它是在单个样本上计算的，而不是您通常理解的、在整个数据集上计算的“信息熵”。
        #
        # 我们来看一个图片输入到克隆模型后会发生什么：
        # 模型会输出一个包含10个概率值的列表，代表它认为这张图片属于10个类别的各自可能性。比如：
        # softmax_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        #
        # 这个概率列表代表了什么？
        # 它代表了模型对这张图片分类的“自信程度”。
        #
        # 什么是“预测熵”？
        # 预测熵就是对这个概率列表计算熵：entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1)。它衡量的是这个概率分布的“混乱程度”或“平坦程度”。
        #
        # 现在我们看两种极端情况：
        #
        # 高不确定性 -> 高熵:
        #
        # 概率分布: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        #
        # 模型的内心独白: “天啊，这张图看起来既像猫又像狗，也像飞机...每个都有10%的可能，我完全懵了，给不出一个明确的判断！”
        #
        # 熵的值: 这个分布非常“平坦”，混乱程度最高，计算出的熵值最大。
        #
        # 结论: 高熵代表了模型的高度不确定性。
        #
        # 低不确定性 (高确定性) -> 低熵:
        #
        # 概率分布: [0.99, 0.001, 0.002, 0.0, ...]
        #
        # 模型的内心独白: “这太明显了，99%的把握就是一架飞机！”
        #
        # 熵的值: 这个分布非常“尖锐”，一点也不混乱，计算出的熵值极小。
        #
        # 结论: 低熵代表了模型的非常确定。
        #
        # 所以，通过计算单个样本的“预测熵”，我们就能精确地量化出模型对这个样本的“困惑程度”。

        #3.我们再看一次核心的权重混合公式：
        # weights = self.alpha * uniform_weights + (1 - self.alpha) * uncertainty_scores
        #
        # 这里的 uncertainty_scores 就是我们刚刚说的预测熵。我们来代入两种情况：
        #
        # 情况一：模型不稳定 -> model_change很高 -> alpha很高 (比如0.9)
        #
        # 公式变成: weights = 0.9 * 随机权重 + 0.1 * 熵
        #
        # 结果: 在最终的采样权重中，随机权重占了主导 (90%)，而熵（不确定性）只占了很小一部分 (10%)。
        #
        # 行动: 采样结果会非常接近于纯粹的随机抽样。
        #
        # 策略: 探索 (Exploration)。当模型自己都不知道该学什么好的时候，就让它广泛地、随机地看新东西。
        #
        # 情况二：模型很稳定 -> model_change很低 -> alpha很低 (比如0.1)
        #
        # 公式变成: weights = 0.1 * 随机权重 + 0.9 * 熵
        #
        # 结果: 在最终的采样权重中，熵（不确定性）占了绝对主导 (90%)。
        #
        # 行动: 采样会极大地倾向于选择那些熵值最高的图片。
        #
        # 策略: 挖掘 (Exploitation)。当模型基础已经打好，就让它集中精力去攻克那些它最搞不懂的“硬骨头”。
        #
        # 总结一下正确的逻辑：
        #
        # alpha 高 -> 侧重随机采样 -> 探索
        #
        # alpha 低 -> 侧重不确定性(高熵)采样 -> 挖掘

        # 重点：所以alpha越低越好，代表模型越稳定对应的公式里面熵就高，选择熵高的图片也就是概率分布很均匀的情况即代表模型高度不确定？怎么感觉这个逻辑这么怪呢？按理说不应该alpha低对应熵低吗？模型越稳定肯定分类越好吧，怎么会对应熵高呢？
        # 您的困惑可以总结为：“一个稳定的好学生（模型稳定），为什么要去专门做难题（选熵高的图片）呢？他不应该做什么题都很自信（熵低）吗？”
        #
        # 答案是：正因为他已经是好学生了，所以才要专门去找难题做，才能从99分提到100分。
        #
        # 让我们一步一步来理清这个逻辑：
        #
        # 第一步：重新理解“模型稳定”（alpha低）
        # 您说的完全正确: 当alpha变低时，代表我们的克隆模型已经学习得不错了，变得很稳定。
        #
        # 这意味着什么？: 这意味着，如果我们现在随机生成1000张图片喂给它，它对其中990张都会给出非常自信的、低熵的预测。比如“99 % 是汽车”，“98 % 是猫”等等。
        #
        # 但是，总有那么10张图片，是它知识的“边缘地带”或“模糊地带”，它看到这些图片时还是会犯迷糊，给出高熵的预测（“10 % 像猫，12 % 像狗，9 % 像卡车...”）。
        #
        # 第二步：确定我们下一步的“学习目标”
        # 既然模型对99 % 的图片都已经很自信了，我们下一步的目标是什么？
        #
        # 错误的目标: 再拿那990张它已经很自信的图片去问老师（查询受害者模型）。这样做是浪费时间和金钱（查询预算），因为这是在重复验证已知知识，无法带来提升。
        #
        # 正确的目标: 从海量图片中，精确地找出那10张模型搞不懂的“疑难杂症”，然后只把这10张图片拿去问老师。这才是最高效的学习方式，是“好钢用在刀刃上”。
        #
        # 第三步：如何执行这个“正确的目标”？
        # 这就是我们那个公式的精妙之处了：
        # weights = alpha * 随机权重 + (1 - alpha) * 熵
        #
        # 当模型稳定，alpha很低时（比如0.1）：
        # weights = 0.1 * 随机权重 + 0.9 * 熵
        #
        # 这个公式在做什么？: 它在下达一个指令：“现在，熵（不确定性）的重要性占90 %！请在生成的候选图片里，优先选择那些熵最高的！”
        #         这个公式的最终目的，就是给那些模型最不确定的（熵最高的）图片，分配一个最大的权重值weights。
        #         为什么权重值大很重要？因为在下一步，代码会调用一个叫做torch.multinomial的函数，这个函数的作用就是“根据权重来进行抽样”。你可以把它想象成一个“智能抽奖机”：
        #         权重越大的奖券，被抽中的概率就越大。
        #         权重越小的奖券，被抽中的概率就越小。
        # 结果是什么？: 算法会无视那990张模型已经很自信的低熵图片，而专门把那10张模型自己都承认“我搞不懂”的高熵图片挑出来。
        self.register_buffer("model_change", torch.tensor(1.0))  # 初始模型变化度量为1.0
        self.register_buffer("alpha", torch.tensor(0.5))  # 初始混合系数为0.5
        self.previous_predictions = None  # 存储上一次迭代的预测

    #创建用于对比学习的特征提取器
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
            nn.Linear(in_features, 128)  # 投影到128维特征空间  通过这个额外的非线性投影，模型被鼓励去放大那些对于“区分不同类别”至关重要的核心语义信息，同时抑制那些无关的细节信息。这个128维的空间是一个专门为了“对比”而优化的空间，在这里，同类样本的特征会聚集得更紧密，不同类样本的特征会分离得更清晰
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

    #计算对比学习的损失
    def contrastive_loss(self, features, labels=None):
        """
        计算对比学习损失
        """
        # 它的目标是教会模型如何学习到更有区分度的特征，而不仅仅是完成分类任务。
        #
        # 这个函数设计得非常巧妙，它可以同时处理两种情况：监督对比学习和自监督对比学习。
        features = nn.functional.normalize(features, dim=1)#作用：对输入的特征向量进行L2归一化。 目的：这是对比学习的标准操作。归一化后，所有特征向量的长度都为1，它们都分布在一个超球面上。这样做的好处是，特征间的相似度可以直接通过**点积（dot product）**来衡量，点积越大，代表两个向量在方向上越接近，即越相似。
        batch_size = features.size(0)

        # 计算特征之间的相似度
        sim_matrix = torch.matmul(features, features.T) / self.temperature #作用：计算一个“相似度矩阵”。torch.matmul(features, features.T)：计算批次内所有特征两两之间的点积，得到一个 batch_size x batch_size 的矩阵。矩阵中 (i, j) 位置的值，就是第 i 个样本和第 j 个样本特征的相似度。/ self.temperature：除以一个“温度系数”。这是一个超参数，可以控制相似度分布的“尖锐”程度。温度越低，会让模型更关注那些与自己非常相似的样本，使得相似度分布更“尖”；温度越高，则会让相似度分布更“平滑”。


        # 创建标签掩码，用于排除自身
        mask = torch.eye(batch_size, device=features.device) #作用：创建一个单位矩阵（对角线上是1，其余是0）。目的：在计算损失时，一个样本永远不应该和它自己进行比较。这个 mask 就是用来在后续计算中方便地“屏蔽”掉对角线上的自身相似度。

        if labels is not None:
            # 监督对比学习 - 相同类别的样本被视为正样本
            pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() #作用：创建一个“正样本掩码”。目的：找出在当前批次中，哪些样本属于同一类别。如果第 i 个样本和第 j 个样本的标签相同，那么 pos_mask 矩阵中 (i, j) 位置的值就为1，否则为0。
            # 排除对角线(自身)
            pos_mask = pos_mask - mask
            # 确保每个样本至少有一个正样本(避免除零错误)
            pos_samples = torch.maximum(pos_mask.sum(1), torch.ones(batch_size, device=features.device))

            # 计算正样本对的损失
            log_prob = sim_matrix - torch.log(torch.exp(sim_matrix).sum(1, keepdim=True) + 1e-10) #作用：这是InfoNCE损失（一种经典的对比学习损失）的核心计算式。它计算了每个样本与所有其他样本的相似度，并将其转换为一个对数概率的形式。

            # 计算每个锚点的监督对比损失
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_samples #作用：计算核心损失。pos_mask * log_prob：利用正样本掩码，只保留那些属于同类别样本对之间的损失值。#.sum(1) / pos_samples：对于每个样本（每一行），将它与所有同类样本的损失加起来并取平均。

            #最小化损失函数其实就是最大化mean_log_prob_pos
            loss = -mean_log_prob_pos.mean() #取整个批次的平均值，并加上负号。
            #监督学习的整体思想是：最大化 mean_log_prob_pos，也就是让每个样本与其所有“正样本”（同类别样本）的特征相似度尽可能高。 这就是“拉近同类”的数学实现。
        else:
            # 自监督对比学习 在没有标签的情况下，我们认为除了样本自己，其他所有样本都是“负样本”。
            log_prob = sim_matrix - torch.log(torch.exp(sim_matrix).sum(1, keepdim=True) + 1e-10)

            # 对角线掩码，排除自身 作用：创建一个“负样本掩码”。它就是一个除了对角线是0，其余全是1的矩阵。
            neg_mask = 1.0 - mask

            # 计算对比损失 计算损失。对于每个样本，我们希望它和所有其他“负样本”的相似度都尽可能低。
            mean_log_prob_neg = (neg_mask * log_prob).sum(1) / (batch_size - 1)
            loss = -mean_log_prob_neg.mean()
            #自监督学习的整体思想是：对于一个样本，将它与自己通过数据增强得到的“副本”视为正样本对（这里未体现，但在更标准的SimCLR等方法中是核心），将其余所有样本都视为负样本。 最终目标是“拉近自己和自己的副本，推远自己和别人”。

        return loss

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
        save_path = os.path.join(self.output_dir, 'class_distribution.png')
        plt.savefig(save_path)
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

        # 存储当前预测用于计算模型变化
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.clone_model(inputs)
                _, predicted = outputs.max(1)
                all_predictions.append(predicted)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"克隆模型测试准确率: {accuracy:.2f}%")

        #为什么在评估的时候要更新awus的参数呢？
        # 简单来说，答案是：因为评估阶段是唯一能够可靠地衡量“模型学习进展”的时机，而AWUS参数的更新恰恰需要依赖这个“进展”信息来动态调整后续的训练策略。
        #
        # 下面我为您详细分解这个逻辑：
        #
        # 1.
        # AWUS的核心：根据模型状态调整策略
        # AWUS的目标是在“随机探索”和“精准挖掘”之间取得平衡。
        #
        # 模型不稳定时（训练初期）：需要进行广泛的随机探索，以快速了解目标模型的整体决策范围。
        #
        # 模型稳定时（训练后期）：需要针对性地在决策边界进行精准挖掘（不确定性采样），以提升窃取效果。
        #
        # 关键问题是，算法如何知道模型当前是“不稳定”还是“稳定”？这就需要一个衡量指标。
        #
        # 2. “模型变化度量”：衡量进展的标尺
        # 您的代码引入了一个非常聪明的指标self.model_change，它的计算方法是：
        #
        # 比较本次评估和上一次评估时，克隆模型在整个测试集上的预测结果有多大的差异。
        #
        # 差异大(model_change高)：说明从上次评估到现在，模型学到了很多新东西，发生了剧烈变化，仍处于“不稳定”期。
        #
        # 差异小(model_change低)：说明模型的预测结果趋于收敛，学习速度放缓，进入了“稳定”期。
        #
        # 3.
        # 为什么这个指标必须在评估（evaluate_clone）时计算？
        # 这是最核心的一点。为了得到一个公平、稳定、有意义的“模型变化”信号，我们必须在两个时间点上，用完全相同的一批数据来进行测试。在您的代码中，这个固定的数据集就是CIFAR - 10的测试集(self.testloader)。
        #
        # 如果在训练时计算：训练时用的都是GAN实时生成的、每一批都不同的随机图片。用这些随机图片来计算“模型变化”是毫无意义的，因为你分不清到底是模型变了，还是仅仅因为这次的输入图片和上次不一样。
        #
        # 在评估时计算：每次调用evaluate_clone函数，模型都会在同一个测试集上完整地跑一遍。这就像用同一把尺子去测量两次身高，得到的差值才是真实的身高变化。因此，只有在这个阶段计算model_change，才能准确地反映出模型自身的学习进展。
        #
        # 计算模型变化
        all_predictions = torch.cat(all_predictions)
        if self.previous_predictions is not None:
            # 计算预测变化的比例
            changes = (all_predictions != self.previous_predictions).float().mean()
            self.model_change = changes
            print(f"模型变化度量: {changes.item():.4f}")

            # 更新AWUS的alpha参数
            self._update_alpha()

        self.previous_predictions = all_predictions

        # 更新最佳模型
        if accuracy > self.best_clone_acc:
            self.best_clone_acc = accuracy
            save_path = os.path.join(self.output_dir, 'best_clone_model.pth')
            torch.save(self.clone_model.state_dict(), save_path)
            print(f"保存新的最佳模型至 {save_path}，准确率: {accuracy:.2f}%")
        return accuracy

    # 这个函数就是我们讨论的“第二步：决策采样策略”。它在每次评估模型后被调用，核心任务是根据模型当前的学习状态，更新下一步的行动方针 alpha。
    # 它和我们之前说的怎么对应上？
    #
    # 输入: 它依赖于self.model_change这个值，这个值是在evaluate_clone函数中计算好的（也就是“感知”步骤的结果）。
    #
    # 决策: 它根据model_change的大小，计算出一个目标alpha值。
    #
    # 平滑处理: 它没有直接使用这个目标alpha，而是通过“指数移动平均”公式，对self.alpha进行一次平滑、稳定的更新。
    #
    # 输出: 更新后的self.alpha。这个值就是我们为下一步采样行动定下的总方针（是该多“探索”还是多“挖掘”）。
    def _update_alpha(self):
        """根据模型变化更新AWUS的alpha混合系数"""
        # 当模型变化大时，增加alpha（更多随机采样）
        # 当模型变化小时，减小alpha（更多不确定性采样）
        # 将“模型变化度”限制在0.1到0.9之间，防止极端值。
        target_alpha = torch.clamp(self.model_change, 0.1, 0.9)
        # 使用“指数移动平均”平滑地更新alpha，而不是让它剧烈跳变。
        self.alpha = self.alpha * self.alpha_decay + target_alpha * (1 - self.alpha_decay)
        print(f"AWUS采样混合系数alpha: {self.alpha.item():.4f}")

    # 这个函数是我们讨论的“第三步：执行智能行动”中的核心计算部分。它接收“决策中心”定下的方针 alpha 和当前这批候选图片的“不确定性分数”，然后计算出最终的抽奖概率。
    # 它和我们之前说的怎么对应上？
    #
    # 它完美地实现了我们之前讨论的权重混合公式：最终权重 = alpha * 随机权重 + (1 - alpha) * 熵。
    # 当 alpha很高时，这个公式的计算结果主要由uniform_weights（随机权重）决定。
    # 当alpha很低时，这个公式的计算结果主要由uncertainty_scores（熵）决定。
    # 它的输出weights就是一张包含了最终抽奖概率的“奖券列表”。
    def get_awus_sampling_weights(self, batch_size, uncertainty_scores=None):
        """
        获取AWUS自适应加权不确定性采样的权重

        参数:
            batch_size: 批次大小
            uncertainty_scores: 如果提供，则使用这些不确定性分数；否则默认为均匀分布

        返回:
            采样权重
        """
        if uncertainty_scores is None:
            # 如果没有不确定性分数，就用均匀权重（纯随机采样）
            weights = torch.ones(batch_size, device=device)
        else:
            # 这就是我们讨论的核心公式！
            # uniform_weights 代表了“探索”的部分。
            uniform_weights = torch.ones_like(uncertainty_scores)
            # 将“探索”和“挖掘”（uncertainty_scores）根据 alpha 进行混合。
            weights = self.alpha * uniform_weights + (1 - self.alpha) * uncertainty_scores

        # 归一化权重，确保所有权重加起来等于1，变成一个合法的概率分布。
        weights = weights / weights.sum()
        return weights

    # 这个函数是“第三步：执行智能行动”的完整流程。它负责生成图片、评估不确定性、计算采样权重，并最终挑出最有价值的样本。
    # 它和我们之前说的怎么对应上？
    #
    # 生成候选: 它首先生成一批备选的图片。
    #
    # 计算熵: 它精确地实现了我们说的“预测熵”的计算，把每张候选图片的“不确定性”量化了出来。
    #
    # 调用计算器: 它把自己算好的“熵”和全局的决策方针alpha一起交给get_awus_sampling_weights函数，拿到了最终的“奖券列表”。
    #
    # 执行抽样: 它使用torch.multinomial这个抽奖机，完成了最终的“择优录取”。
    #
    # 最终输出: 返回被精心挑选出来的、对模型学习最有价值的一批图片selected_images。
    def generate_samples_with_awus(self, batch_size, z=None):
        """
        使用AWUS策略生成样本

        参数:
            batch_size: 批次大小
            z: 可选的预定义噪声向量

        返回:
            生成的图像和对应的不确定性权重
        """
        # 生成一批“候选”图片（通常数量会比最终需要的多，比如2倍）。
        if z is None:
            z = torch.randn(batch_size * 2, self.nz, device=device)  # 生成额外的样本以供选择

        # 生成候选图像
        with torch.no_grad():
            candidate_images = self.generator(z)
            candidate_images = self._ensure_three_channels(candidate_images)

            # 使用克隆模型，计算这批候选图片的“不确定性”，也就是“预测熵”。
            outputs = self.clone_model(candidate_images)
            softmax_probs = torch.softmax(outputs, dim=1)

            # 计算预测熵作为不确定性度量
            entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1)

            # 获取AWUS采样权重。调用“行动策略计算器”，根据当前的alpha和刚算出的熵，获取最终的采样权重。
            weights = self.get_awus_sampling_weights(len(entropy), entropy)

            # 使用“智能抽奖机”(torch.multinomial)，根据权重列表进行抽奖，
            # 选出最终的一批图片。
            indices = torch.multinomial(weights, batch_size, replacement=False)

        # 拿出被选中的图片和它们对应的权重。
        selected_images = candidate_images[indices]
        selected_weights = weights[indices]

        return selected_images, selected_weights

    def train(self, num_queries=8000000, batch_size=128, g_steps=1, c_steps=1, evaluate_every=100000):
        """
        训练克隆模型，实现增强版DFMS-HL算法
        """
        print("开始SwiftThief-AWUS增强版DFMS-HL训练...")

        # 重置查询计数器
        initial_queries = self.query_count
        real_label = 1
        fake_label = 0

        # 保存训练历史
        history = {
            'g_loss': [], 'div_loss': [], 'adv_loss': [], 'd_loss': [],
            'c_loss': [], 'contrast_loss': [], 'query_count': [], 'accuracy': [],
            'alpha': [], 'model_change': []
        }

        # 可视化初始类别分布
        print("初始类别分布:")
        self.visualize_class_distribution()

        # 创建一个进度条
        pbar = tqdm(total=num_queries, desc="SwiftThief-AWUS训练")
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
                # 使用AWUS策略生成样本
                fake_images, _ = self.generate_samples_with_awus(batch_size)

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
                # 使用AWUS策略生成新样本
                fake_images, _ = self.generate_samples_with_awus(batch_size)

                # 从目标模型获取硬标签
                hard_labels = self.get_hard_label(fake_images)

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

                loss = self.criterion_ce(outputs, hard_labels)
                c_loss_sum += loss.item()

                # 总损失 = 原始损失 + 对比学习损失
                total_loss = loss + self.lambda_contrast * contrast_loss

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
            contrast_loss_avg = contrast_loss_sum / c_steps

            history['g_loss'].append(g_loss_avg)
            history['div_loss'].append(div_loss_avg)
            history['adv_loss'].append(adv_loss_avg)
            history['d_loss'].append(d_loss_avg)
            history['contrast_loss'].append(contrast_loss_avg)
            history['query_count'].append(self.query_count)
            history['alpha'].append(self.alpha.item())
            history['model_change'].append(self.model_change.item())

            # 更新进度条
            new_queries = self.query_count - initial_queries - pbar.n
            pbar.update(new_queries)
            pbar.set_postfix(g_loss=f"{g_loss_avg:.4f}",
                             contrast=f"{contrast_loss_avg:.4f}",
                             alpha=f"{self.alpha.item():.4f}",
                             queries=self.query_count)

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
        final_model_path = os.path.join(self.output_dir, 'clone_model_final.pth')
        torch.save(self.clone_model.state_dict(), final_model_path)

        final_extractor_path = os.path.join(self.output_dir, 'feature_extractor_final.pth')
        torch.save(self.feature_extractor.state_dict(), final_extractor_path)
        self._save_training_history(history)

        print(f"SwiftThief-AWUS训练完成，总查询数: {self.query_count}")
        print(f"最佳克隆模型准确率: {self.best_clone_acc:.2f}%")

        # 绘制训练损失
        self._plot_training_history(history)

        # 最终评估
        final_acc = self.evaluate_clone()
        print(f"最终克隆模型准确率: {final_acc:.2f}%")

        return history

    # 初始化克隆模型，使用对比学习和AWUS
    def initialize_clone_with_contrastive(self, num_iterations=50000, batch_size=128, evaluate_every=10000):
        """初始化克隆模型，使用对比学习和AWUS策略"""
        print("初始化克隆模型...")
        self.clone_model.train()
        self.feature_extractor.train()

        # 创建一个进度条以便于跟踪
        pbar = tqdm(total=num_iterations, desc="初始化克隆模型")

        for i in range(num_iterations):
            # 使用AWUS生成样本
            fake_images, _ = self.generate_samples_with_awus(batch_size)

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
            # self.scheduler_C.step()
            # self.scheduler_F.step()

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
        history_path = os.path.join(self.output_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

    def _plot_training_history(self, history):
        """绘制训练历史图表"""
        plt.figure(figsize=(18, 15))

        # 绘制生成器损失
        plt.subplot(4, 2, 1)
        plt.plot(history['query_count'], history['g_loss'])
        plt.title('生成器损失')
        plt.xlabel('查询次数')

        # 绘制判别器损失
        plt.subplot(4, 2, 2)
        plt.plot(history['query_count'], history['d_loss'])
        plt.title('判别器损失')
        plt.xlabel('查询次数')

        # 绘制多样性损失
        plt.subplot(4, 2, 3)
        plt.plot(history['query_count'], history['div_loss'])
        plt.title('多样性损失')
        plt.xlabel('查询次数')

        # 绘制对抗损失
        plt.subplot(4, 2, 4)
        plt.plot(history['query_count'], history['adv_loss'])
        plt.title('对抗损失')
        plt.xlabel('查询次数')

        # 绘制对比学习损失
        plt.subplot(4, 2, 5)
        plt.plot(history['query_count'], history['contrast_loss'])
        plt.title('对比学习损失')
        plt.xlabel('查询次数')

        # 绘制AWUS alpha参数
        plt.subplot(4, 2, 6)
        plt.plot(history['query_count'], history['alpha'])
        plt.title('AWUS混合系数Alpha')
        plt.xlabel('查询次数')

        # 绘制模型变化度量
        plt.subplot(4, 2, 7)
        plt.plot(history['query_count'], history['model_change'])
        plt.title('模型变化度量')
        plt.xlabel('查询次数')

        # 绘制准确率
        evaluate_every = history['query_count'][-1] // len(history['accuracy']) if history['accuracy'] else 1
        query_points = [i * evaluate_every for i in range(len(history['accuracy']))]

        plt.subplot(4, 2, 8)
        plt.plot(query_points, history['accuracy'])
        plt.title('克隆准确率')
        plt.xlabel('查询次数')
        plt.ylabel('准确率 (%)')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(save_path)
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
    VICTIM_MODEL_PATH = "results/0_victim_model/victim_model.pth"
    SYNTHETIC_GEN_PATH = "results/1_gan_training/dcgan_generator_synthetic.pth"
    PROXY_GEN_PATH = "results/1_gan_training/dcgan_generator_proxy.pth"
    OUTPUT_DIR = "results/3_innovation_attack"

    # 加载目标模型
    victim_model = load_victim_model(VICTIM_MODEL_PATH)

    if victim_model is None:
        print("无法加载目标模型，请确保文件存在！")
        exit(1)

    # 选择使用哪种预训练生成器
    use_synthetic = False  # 设置为False以使用代理数据集的生成器

    if use_synthetic:
        # 使用合成数据
        proxy_dataset = SyntheticDataset(size=50000)
        generator = load_generator(SYNTHETIC_GEN_PATH, nc=1)
        nc = 1
    else:
        # 使用CIFAR-100代理数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # 选择40个类别
        proxy_dataset = get_proxy_dataset(num_classes=40)
        generator = load_generator(PROXY_GEN_PATH, nc=3)
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

    # 初始化DFMS_HL_Swift_AWUS
    dfms_hl = DFMS_HL_Swift_AWUS(victim_model, generator, proxy_dataset=proxy_dataset, nc=nc,
                                 lambda_div=500, alpha_decay=0.9, output_dir=OUTPUT_DIR)

    # --- 步骤 1: 第一次初始化克隆模型 (得到一个“老师” C1) ---
    print("【步骤1】第一次初始化克隆模型 (用于指导生成器)...")
    dfms_hl.initialize_clone_with_contrastive(num_iterations=20000, evaluate_every=5000)
    # (同样，请确保这里的学习率调度器问题已按之前的建议修复)

    # --- 【关键补充】步骤 2: 从头训练一个新的、最终的克隆模型 ---
    # 您的创新方法将G的训练融合在了主循环里，所以我们在这里需要
    # 重置克隆模型，让它准备好用一个“进化后的G”来学习。

    print("【步骤2 - 新增】重置并重新创建最终的克隆模型...")
    # 重新创建克隆模型、特征提取器以及它们的优化器和调度器
    dfms_hl.clone_model = create_clone_model().to(device)
    dfms_hl.feature_extractor = dfms_hl._create_feature_extractor().to(device)

    dfms_hl.optimizer_C = optim.SGD(dfms_hl.clone_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    dfms_hl.optimizer_F = optim.SGD(dfms_hl.feature_extractor.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    dfms_hl.scheduler_C = CosineAnnealingLR(dfms_hl.optimizer_C, T_max=200)
    dfms_hl.scheduler_F = CosineAnnealingLR(dfms_hl.optimizer_F, T_max=200)

    # 再次调用初始化函数，用一个已经有初步指导能力的G来训练我们最终的C
    # 这一步相当于用“稍微好一点的教材”来教出一个“基础更好的学生”
    print("【步骤3】再次初始化，训练最终克隆模型...")
    dfms_hl.initialize_clone_with_contrastive(num_iterations=20000, evaluate_every=5000)

    # --- 步骤 4: 开始最终的交替训练 ---
    print("【步骤4】开始最终的交替训练...")
    history = dfms_hl.train(
        num_queries=3000000,  # 注意调整查询预算
        batch_size=128,
        g_steps=1,
        c_steps=1,
        evaluate_every=100000
    )
