import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#预训练数据生成器

# 文件2只是在打基础，用最抽象的东西去打基础，相当于预训练，打好基础后文件3再去指定具体生成任务，这个抽象的东西在DCGan里面就是“正品"
# 打基础 (文件2)：就像教一个孩子认识最基本的笔画（点、横、竖、撇、捺），而不急于教他写具体的字。我们用随机几何图形（最抽象的东西）来训练GAN，就是为了让它掌握生成图像最底层的“笔画”能力，这就是一个通用的、无偏见的预训练过程。
# 指定具体任务 (文件3)：基础打好后，我们就可以给这个孩子布置各种具体的“书法作业”了。在文件3中，我们就对这个已经会“画笔画”的GAN下达指令，比如：“请你写一个最能让A老师（克隆模型）困惑的字（类多样性损失）。”“请你写一个B老师（目标模型）最难分辨的字（边界探测）。”...等等

# 在我们的模型窃取攻击计划中，我们需要用海量的、各种各样的数据去反复查询、试探我们想要窃取的目标模型（Victim Model）。但现实中，攻击者往往没有目标模型当初所用的真实训练数据（比如CIFAR-10）。
#
# 那么，用来查询的大量数据从哪里来呢？—— 自己造！
#
# 这个 2_train_gan.py 文件就是用来训练一个生成对抗网络 (Generative Adversarial Network, GAN)。这个GAN学会之后，就能源源不断地为我们生成以假乱真的图片，这些图片将作为“子弹”，在第三阶段的攻击脚本中去“射击”我们的目标模型。


# 这个脚本通过训练一个叫做 DCGAN (深度卷积生成对抗网络) 的模型来实现图片生成。DCGAN内部有两个角色，它们就像“矛”和“盾”一样，在互相的对抗竞争中共同进步：
#
# 生成器 (Generator)：一个“伪造者”。它接收一串随机数字（噪声），然后努力把这串数字变成一张看起来很真实的图片。
#
# 判别器 (Discriminator)：一个“鉴宝师”。它负责判断一张图片是“真品”（来自真实数据集）还是“赝品”（由生成器伪造的）。
#
# 训练过程：
#
# 生成器不断尝试造假，希望能骗过判别器。
#
# 判别器不断学习，希望能精准地识别出生成器的伪作。
#
# 经过成千上万轮的“道高一尺，魔高一丈”的对抗后，生成器造假的能力会变得越来越强，最终能生成出非常逼真的图片。
#
# 这个脚本非常巧妙，它模拟了两种不同资源情况下的攻击者，并分别为他们训练了两种不同的“伪造工厂”：
#
# 模式一：零知识攻击者
#
# 情况：攻击者手上没有任何和目标任务相关的真实图片。
#
# 对策：脚本中使用 SyntheticDataset 类来生成完全是随机几何形状的合成图片。然后用这些合成图片去训练GAN。
#
# 产出：dcgan_generator_synthetic.pth，一个学会了生成几何图形的生成器。
#
# 模式二：有“代理数据”的攻击者
#
# 情况：攻击者手上有一些和目标任务不完全一样，但有点相关的图片。比如，目标API是识别10种常见动物（CIFAR-10），而攻击者手上有另一个包含100种物品的数据集（CIFAR-100）。
#
# 对策：脚本中使用 get_proxy_dataset 函数，从CIFAR-100中随机抽取一部分数据，作为“代理数据”。然后用这些代理数据去训练GAN。
#
# 产出：dcgan_generator_proxy.pth，一个学会了生成更像真实世界物体的生成器。

# 这体现了研究的科学严谨性。通过设置这两种模式，作者实际上是在设计一个对照实验，目的是为了回答一个更深层次的问题：
#
# “攻击者的初始数据质量，对模型窃取攻击的最终效果有多大影响？”
#
# 模式一（合成数据）是实验的“底线”或“下限” (Worst-Case Baseline)。它代表了最苛刻、最困难的攻击条件。如果在这种“一无所有”的情况下，攻击都能成功，那说明攻击方法本身非常强大。
#
# 模式二（代理数据）则更加贴近真实世界中的攻击场景。在现实中，攻击者往往能从网络上爬取到一些与目标任务类似的数据。例如，想攻击一个猫狗识别器，攻击者可能手头有一个包含各种动物的公开数据集。
# 后续所有核心的攻击脚本，在默认设置下，使用的都是在“模式二”（代理数据）下训练出的生成器，也就是 dcgan_generator_proxy.pth。代码的灵活性也允许研究者随时将 use_synthetic 改为 True，来测试在最差数据条件下，这些先进的攻击策略表现如何。
# 在GAN的训练世界里，从dataloader里取出的数据，就是它要学习模仿的“真品”和“标准”。无论是随机几何图形，还是CIFAR-100的代理图片，在那一刻，它们就是GAN需要对标的“黄金标准”。
#
# “此时训练的DCGan也是预训练？”： 是的！这正是点睛之笔！ 在我们这个模型窃取攻击的大项目中，训练DCGAN这个过程，它本身就是一种“预训练” (Pre-training)。
#
# 为什么是“预训练”？ 因为我们训练这个GAN，并不是为了得到一个能画画的AI拿去参加艺术展。它的最终目的，是为了在接下来的文件3中，作为一个高效的数据生成工具来使用。
#
# 它预训练了什么？ 它预训练了一个生成器（Generator），让这个生成器掌握了“从无到有生成多样化、有结构图片”的通用能力。
# 文件2（当前文件）：这是准备阶段。我们利用手头有的、攻击者能接触到的数据（几何图形或代理数据）作为“真品”，来训练出一个强大的生成器G。在这个阶段，判别器D和这些“真品”都是一次性的工具，是为了把G“陪练”出来。
#
# 阶段一 (文件2): 预训练。我们使用“真品”教科书（代理数据的dataloader）同时训练一个生成器G和一个判别器D_pretrain。训练结束后，保存G的权重。
#
# 阶段二 (文件3): 攻击。
#
# 准备：加载阶段一训练好的G的权重。扔掉D_pretrain。保留“真品”教科书(dataloader)。创建一个全新的判别器D_new并初始化。
#
# 交替训练循环：
#
# 训练D_new: 同时使用“真品”教科书中的真实图片 和 G生成的假图片来训练D_new。
#
# 训练G: 使用D_new的反馈（对抗损失）和克隆模型C的反馈（类多样性损失）来更新G。
#
# 训练C: 使用G生成的假图片去查询受害者模型，用得到的标签来训练C。


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "results/1_gan_training"
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# --- 数据集定义 ---

# 这是一个自定义的数据集类，完全复现了论文中提到的“合成数据”
# 具体做法：我们的目标模型（victim_model）是用来识别真实世界物体的（飞机、汽车、鸟等）。那么，什么东西和这些真实物体差异最大呢？答案就是抽象的、随机的几何图形。
#
# 目的：如果一个攻击方法，在只能用这些风马牛不相及的“乱码”图片作为初始弹药的情况下，最后依然能成功窃取出能识别真实物体的模型，那就足以证明这个攻击算法本身具有极强的能力，它不依赖于高质量的初始数据。这为算法的性能设置了一个非常可信的“下限”。
# 完成模式一：零知识攻击者
class SyntheticDataset(Dataset):
    """创建合成数据集，包含随机形状（三角形、矩形、椭圆或圆形）"""

    def __init__(self, size=50000, image_size=32):
        self.size = size # 要生成多少张图片
        self.image_size = image_size # 图片尺寸
        self.data = []

        # 在初始化时，就循环生成所有图片并保存在内存里
        for _ in range(size):
            img = self.generate_random_image()
            self.data.append(img)

    def generate_random_image(self):
        # 1. 创建随机背景色的画布
        bg_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

        # 创建图像
        img = Image.new('RGB', (self.image_size, self.image_size), bg_color)
        draw = ImageDraw.Draw(img)# 创建一个可以在上面绘画的对象

        # 2. 在画布上画1到3个随机形状
        for _ in range(random.randint(1, 3)):
            # 选择形状类型
            shape_type = random.choice(['triangle', 'rectangle', 'ellipse', 'circle'])

            # 随机形状颜色
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

            # 随机定义形状的位置和大小
            x1 = random.randint(0, self.image_size - 1)
            y1 = random.randint(0, self.image_size - 1)
            size = random.randint(5, 20)
            x2 = min(x1 + size, self.image_size - 1)
            y2 = min(y1 + size, self.image_size - 1)

            # 根据类型画出形状
            if shape_type == 'triangle':
                # 三角形的三个点
                points = [
                    (x1, y1),
                    (x2, y1),
                    ((x1 + x2) // 2, y2)
                ]
                draw.polygon(points, fill=color)
            # ... (其他形状的绘制逻辑，这里省略，但原理相同)
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

        # 3. 按照论文的描述，将彩色图片转为灰度图
        img = img.convert('L')
        ## 4. 转换为Tensor并归一化到[-1, 1]范围，这是GAN的常见做法
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5,), (0.5,))(img)  # 标准化到[-1, 1]
        return img

    def __len__(self):
        return self.size # 返回数据集的总长度

    def __getitem__(self, idx):
        return self.data[idx], 0  # 返回指定索引的图片。标签在这里不重要，随便给一个0。

# 完成模式二：有代理数据的攻击者
# 定义一个函数，用于获取CIFAR-100的一部分作为代理数据
def get_proxy_dataset(num_classes=40):
    """获取CIFAR-100的子集作为代理数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化到[-1, 1]
    ])

    cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # 从100个类别中，随机挑选40个类别
    selected_classes = random.sample(range(100), num_classes)
    # 找到所有属于这40个类别的图片的索引
    indices = [i for i, (_, label) in enumerate(cifar100) if label in selected_classes]

    # 使用Subset创建一个只包含这些索引的新数据集
    # 创建子集
    subset = torch.utils.data.Subset(cifar100, indices)
    return subset


# --- DCGAN 模型定义 ---
# 这部分是DCGAN的经典实现，是一个非常标准的“代码模板”。
# 定义DCGAN生成器和判别器
# 定义生成器，生成器的任务是把一串毫无意义的随机数字（噪声，nz），变成一张有结构的图片（nc个通道）。它是一个“放大”的过程。
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        """
        参数：
        # nz: 输入噪声向量的维度 (latent vector size)
        # ngf: 生成器内部特征图的深度 (generator feature map size)
        # nc: 输出图片的通道数 (number of channels)
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 这是一个“反卷积”（更准确说是转置卷积）层，作用和卷积相反，用于将小的特征图放大。
            # 输入是 nz x 1 x 1 的噪声
            # 输入是nz维的随机噪声，输出是ngf*4的特征图
            #工序1：nn.ConvTranspose2d(...) -> nn.BatchNorm2d -> nn.ReLU
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),#这是核心部件，叫做转置卷积（常被误称为反卷积）。它的作用和普通卷积正好相反，不是把图像变小提取特征，而是把小的特征图放大。您可以想象成把一小块像素，智能地“扩散”成一大块。这是第一层，它把一个1x1大小的噪声，直接放大成一个4x4的特征图。
            nn.BatchNorm2d(ngf * 8),#批量归一化。这是GAN训练中一个至关重要的“稳定器”。它能防止梯度消失或爆炸，让数据在网络中流动得更顺畅，极大地稳定了GAN的训练过程。
            nn.ReLU(True),#修正线性单元。一个激活函数，它给网络引入了非线性，使得网络能够学习更复杂的模式。
            #接下来的几组工序，结构完全一样，都是 ConvTranspose2d -> BatchNorm2d -> ReLU。它们在做同一件事：将特征图的尺寸翻倍，同时减少通道数（特征的深度）。
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
            # 论文中的DCGAN生成64x64的图，这里做了一些适配
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()# Tanh激活函数将输出值缩放到[-1, 1]之间，正好匹配我们数据的归一化范围。这是最后的激活函数。Tanh函数会将所有输出值压缩到 -1到1 的范围内。这非常关键，因为我们之前在数据预处理时，也把图片的像素值归一化到了-1到1的范围。这样一来，生成器“伪造”出的图片，其数值范围就和真实图片完全一致了，这对判别器公平地进行比较至关重要。
            #nn.Tanh() 函数就像是最后一道“上釉”或“上色”的工序。它把泥人身上所有点的数值（像素值）都严格地规范到 -1 到 1 的范围内，让它看起来颜色均匀、表面光滑，符合我们最终想要的图片格式。

        )
        # 对于CIFAR-10，我们需要调整输出大小
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)

    def forward(self, input):
        # input的形状是 (batch_size, nz)，需要先reshape成 (batch_size, nz, 1, 1)才能喂给卷积网络
        x = input.view(-1, input.size(1), 1, 1)  # 将输入重塑为4D张量，把一维的噪声向量，变成一个小小的、有深度（通道数nz）、有长和宽（1x1）的“黏土块”。
        x = self.main(x)#送入流水线，开始塑形 (调用 self.main),它们像接力赛一样，一棒接一棒地将一个微小的噪声点，逐步“吹气”放大，并刻画出细节，最终形成一张完整的图片。
        return self.upsample(x)  # 调整为32x32大小


# 定义判别器
# 浓缩信息，最终打分.它接收一张复杂的、高维度的图片（比如3x32x32），然后通过一系列操作，最终把它压缩成一个单一的数字（一个0到1之间的概率值）。
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        """
        参数：
        # ndf: 判别器内部特征图的深度
        # nc: 输入图片的通道数
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 和生成器相反，判别器使用标准的卷积层，将大图一步步缩小。
            # 输入是(nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图大小: (ndf) x 16 x 16
            # 接下来的几组工序，结构几乎完全一样，都是
            # Conv2d -> BatchNorm2d -> LeakyReLU。它们在做同一件事：不断地将特征图的尺寸减半，同时将特征的深度（通道数）翻倍。
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),# LeakyReLU是GAN中常用的激活函数
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
            nn.Sigmoid() # Sigmoid激活函数将输出压缩到(0, 1)之间，正好可以表示“图片为真的概率”。
        )

    def forward(self, input):
        # squeeze(1)将形状从 (batch_size, 1) 变为 (batch_size,)
        return self.main(input).view(-1, 1).squeeze(1)


# --- 训练函数 ---
# 训练DCGAN
def train_dcgan(dataloader, synthetic=False, num_epochs=50, save_path='dcgan_generator.pth'):
    """预训练DCGAN生成器"""
    # 创建生成器和判别器
    nz = 100  # 潜在向量的大小
    # 如果是合成数据(synthetic=True)，它是单通道灰度图，所以nc=1。否则是3通道彩色图。
    nc = 1 if synthetic else 3  # 通道数：1用于灰度合成图像，3用于彩色CIFAR-100

#第一部分：准备阶段 (初始化一切)

    netG = Generator(nz=nz, nc=nc).to(device)
    netD = Discriminator(nc=nc).to(device)

    # 初始化权重，这是DCGAN论文推荐的做法
    def weights_init(m):
        classname = m.__class__.__name__# 获取模块的类名，比如 'Conv2d' 或 'BatchNorm2d'
        # ... 下面是针对不同类别模块的初始化 ...
        # m.weight.data: 要初始化的目标是这个模块的权重数据。
        # 0.0, 0.02: 这是正态分布的两个参数：均值为0，标准差为0.02。
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # 设置优化器
    criterion = nn.BCELoss() # 二元交叉熵损失，用于GAN训练
    fixed_noise = torch.randn(64, nz, device=device)  # 用于可视化 # 一块固定的噪声，用于在训练过程中生成样本，观察G的进步
    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print("开始预训练DCGAN...")

#第二部分：对决循环 (猫鼠游戏正式开始)
    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(dataloader):
            ############################
            # (1) 更新判别器D: 最大化log(D(x)) + log(1 - D(G(z)))
            ###########################
            # 训练使用真实样本
            netD.zero_grad()# 清空梯度

            #第一步：看真品
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
            output = netD(real_data)# 把一批真品给侦探D看
            errD_real = criterion(output, label) # 计算D在真实样本上的损失
            errD_real.backward()

            # 第二步：看赝品
            # 训练使用生成的假样本
            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(noise) # 伪造大师G现场创作一批赝品
            label.fill_(fake_label)
            # .detach() 是关键！它会阻断梯度流向G。我们只想更新D，不想让G在这一步被更新。 它的意思是“把赝品和伪造大师G暂时切断联系”。我们是在训练侦探D，我们只想让D学会识别这批“静态的”赝品，而不希望这次反思的梯度传回到G那里去影响G。就像侦探在分析一幅假画时，他只关心画本身，而不会去影响那个正在隔壁房间作画的伪造者。
            output = netD(fake.detach())
            errD_fake = criterion(output, label) # 计算D在虚假样本上的损失
            errD_fake.backward()

            errD = errD_real + errD_fake # 总损失
            optimizerD.step()  # 更新D的参数

            ############################
            # (2) 更新生成器G: 最大化log(D(G(z)))
            ###########################
            netG.zero_grad()#梯度清空
            label.fill_(real_label)  # G的目标是让D认为它的作品是“真实”的
            output = netD(fake) # 这里没有 .detach()，因为我们需要梯度回传到G
            errG = criterion(output, label) # 计算G的损失
            errG.backward()
            optimizerG.step() # 更新G的参数

            if i % 100 == 0:
                print(
                    f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')



# 我们把训练过程想象成一条正在高速运转的“生产线”。
#
# 主生产线（for i, (data, _) in enumerate(dataloader): 循环）
#
# 在这条主生产线上，每一秒钟，我们都在做同一件事：
#
# 拿来一车新的原材料（noise = torch.randn(...)）。
#
# 命令生成器G用这些新材料，实时地、动态地生产出一批全新的“产品”（fake = netG(noise)）。
#
# 把这批新产品立刻送去给判别器D检验，然后根据检验结果，立刻更新G和D的生产技能。
#
# 这条生产线从未停止，也从不使用旧的产品。每一批用于训练的产品都是即时生产、用完即弃的。
#
# 拍照留念（if epoch % 10 == 0: 代码块）
#
# 每隔一段时间（比如每10个大回合），生产线会暂停一瞬间。
#
# 导演会拿出那块固定不变的“标准原材料” (fixed_noise)，让生成器G用它当前的最高技艺，生产出一批“纪念品”。
#
# 然后，最关键的来了：导演会把这批“纪念品”拿走，拍一张照片（plt.savefig(...)），然后就把这批纪念品扔掉了！
#
# 拍完照后，主生产线立刻恢复运转，继续它“用新材料 -> 造新产品 -> 训练”的循环。第一轮结束后会生成一批图片展示出来，这个生成的图片不会被用在后续训练

        # 每个epoch结束后，用fixed_noise生成图片，看看效果
        # 保存生成的图像
        if epoch % 10 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                # 显示和保存图像
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                if synthetic:
                    # 对于单通道灰度图像
                    # 使用torchvision.utils.make_grid将多张小图拼成一张大图
                    plt.imshow(np.transpose(torchvision.utils.make_grid(
                        fake, normalize=True).cpu(), (1, 2, 0)), cmap='gray')
                else:
                    # 对于彩色图像
                    plt.imshow(np.transpose(torchvision.utils.make_grid(
                        fake, normalize=True).cpu(), (1, 2, 0)))
                sample_save_path = os.path.join(SAMPLES_DIR,
                                                f'dcgan_samples_{"synthetic" if synthetic else "proxy"}_epoch_{epoch}.png')
                plt.savefig(sample_save_path)
                plt.close()

    # 训练结束后保存生成器的权重
    torch.save(netG.state_dict(), save_path)
    print(f'生成器已保存至 {save_path}')
    return netG, netD


# 主函数
if __name__ == "__main__":
    # 1. 准备两种数据集
    synthetic_dataset = SyntheticDataset(size=50000)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=128, shuffle=True, num_workers=2)

    # 使用CIFAR-100子集作为代理数据
    proxy_dataset = get_proxy_dataset(num_classes=40)
    proxy_loader = DataLoader(proxy_dataset, batch_size=128, shuffle=True, num_workers=2)

    # 2. 分别训练两个生成器
    # 训练使用合成数据的DCGAN
    synthetic_gen_path = os.path.join(OUTPUT_DIR, "dcgan_generator_synthetic.pth")
    train_dcgan(synthetic_loader, synthetic=True, num_epochs=50, save_path=synthetic_gen_path)

    # 训练使用代理数据的DCGAN
    proxy_gen_path = os.path.join(OUTPUT_DIR, "dcgan_generator_proxy.pth")
    train_dcgan(proxy_loader, synthetic=False, num_epochs=50, save_path=proxy_gen_path)
