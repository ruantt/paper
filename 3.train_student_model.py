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
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#您的真正目的是生成一个全新的、专门用于攻击的数据集，这个数据集能让克隆模型C以最快的速度学会受害者模型V的分类逻辑。生成的这张抽象图片，它本身没有任何标签。它不是猫，也不是狗。它到底是什么类别，完全由受害者模型V来定义。您定义的10个类别，实际上是克隆模型C要去模仿的受害者模型V的10种分类行为。
# 我们无法得到那本加密的私有数据集，但我们知道秘籍是用什么网络架构写的。于是，我们找来一个GAN，让他不断地画出各种假的生成的图片，然后我们拿着这些假图谱去请教受害者模型，查询API获取标签。
# 宗师的每一次点头或摇头，都在帮助我们拼凑出那本失传秘籍的完整内容（窃取权重）。
#目标不同: 传统GAN的目标是**“模仿”，即生成与真实数据集（如CIFAR-10）在视觉上无法区分的样本。而本框架的目标是“审问”**（Interrogation），即生成能够最高效揭示受害者模型（V）内部决策逻辑的查询样本。这些样本无需逼真，只需具备信息量。

#我这个和普通的GAN不一样是因为普通的在判别器判别时真样本就是实际的数据集在我这里就是所谓的cifair10然后看生成器生成的样本和cifair10的差距计算loss然后更新，此时目的是让生成器生成的样本更接近真实样本。但是我这个情况属于是我拿不到cifair10我只能随便制造点数据集当做真样本那么此时我的输出不再是输出接近正样本的数据集而是生成探测边界，让克隆模型C以最快的速度学会受害者模型V的分类逻辑
# “真样本”的角色转变: 在此框架中，我们无法访问真实的训练数据。因此，任意选择的代理数据集（如合成形状）的角色不再是模仿的终点，而仅仅是作为一个**“图像结构正则化器”**。它通过对抗损失，确保生成器（G）的输出具备图片的基本属性（如边缘、纹理），而非无意义的噪声，即教会G“如何画”，而非“画什么”。
#
# 标签的定义者: 生成的探针样本（可能是抽象的、无意义的图形）本身没有类别。它的标签完全由受害者模型V的输出来定义。当V对一个探针预测为“猫”，那么这个探针的“真实”标签就被定义为“猫”。
# 基线攻击方法

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
# 在这里，model_path 的默认值被设为了 'dcgan_generator_synthetic.pth'。这表示：如果调用 load_generator 函数时不给它指定一个路径，它就会默认去加载这个合成数据训练的生成器。
def load_generator(model_path, nc=1):
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
    def __init__(self, victim_model, generator, proxy_dataset=None, nc=3, nz=100, lambda_div=500,output_dir="results/2_baseline_attack"):
        """
        初始化DFMS-HL算法

        参数:
            victim_model: 目标模型
            generator: 预训练的生成器
            nc: 图像通道数
            nz: 噪声向量维度
            lambda_div: 类多样性损失的权重系数
        """

        # 它接收已经训练好的受害者模型和生成器。
        #
        # 它创建了两个全新的模型：判别器和克隆模型。这两个模型是从零开始，等待被训练的。保留预训练的“天才选手”G：我们花大力气预训练，就是为了得到一个高起点的生成器。此时当时训练的D不要了。当场任命一位“新裁判”D：在窃取阶段重新创建一个判别器，让它从零开始，与这个进化版的G和新加入的C一起动态地学习和适应。这个新的D能更准确地为当前阶段的G提供梯度，从而更好地指导整个窃取过程。


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
        #更新G的权重
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        #更新D的权重
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        #更新C的权重
        self.optimizer_C = optim.SGD(self.clone_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler_C = CosineAnnealingLR(self.optimizer_C, T_max=200)

        # 最佳模型跟踪
        self.best_clone_acc = 0.0

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建测试数据加载器用于验证
        #加载的是官方的、标准的CIFAR-10测试集。
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=0)

        # 添加代理数据集
        #“代理数据模式”中用到的那个CIFAR-100子集。它在窃取阶段，继续作为“真图片”来训练判别器D。
        self.proxy_dataset = proxy_dataset
        if proxy_dataset is not None:
            self.proxy_loader = DataLoader(proxy_dataset, batch_size=128, shuffle=True, num_workers=0)
            self.proxy_iterator = iter(self.proxy_loader)

    # 对一个已经加载的、预训练好的生成器进行“二次初始化”或者说“微调（Fine-tuning）——预热阶段
    # 简而言之，这个函数在说：
    #
    # “嘿，生成器G，我知道你已经很会画画了。但在我们开始正式攻击之前，你先随便画几笔，让我看看旁边这个新手克隆模型C对你的画有什么反应。根据它的反应，你再稍微调整一下你的画风，好让我们接下来的合作更顺畅。”
    #
    # 在initialize_generator执行完毕后，我们得到的G，就已经不是原来那个只会生成通用图片的G了，而是一个初步适应了当前克ronlone模型C特性的、更有针对性的G。
    def initialize_generator(self, nG=2000, batch_size=128):
        """
        初始化生成器，迭代次数降低到合理水平
        论文可能指的是生成的样本总数而不是迭代次数
        """
        print("预热生成器...")
        self.generator.train()#将生成器G设置为训练模式
        self.discriminator.train()#将判别器D也设置为训练模式

        real_label = 1
        fake_label = 0

        # 预先准备代理数据批次（如果可用）
        #这部分代码是为了提高效率。它会从代理数据集（self.proxy_loader）中预先加载最多100个批次的真实图片，并把它们缓存到内存（proxy_batches）里。这样在后续的循环中，就可以直接从内存中快速取用，而不需要每次都从磁盘读取，从而加速训练。
        proxy_batches = []
        if hasattr(self, 'proxy_loader') and self.proxy_loader is not None:
            for data, _ in self.proxy_loader:
                proxy_batches.append(data.to(device))
                if len(proxy_batches) >= 100:  # 缓存100个批次足够了
                    break

        #进度条设置，这能让我们在运行时直观地看到这个微调过程的进度
        pbar = tqdm(total=nG, desc="初始化生成器")

        for i in range(nG):
            # 生成随机噪声
            z = torch.randn(batch_size, self.nz, device=device)

            # 生成假图像
            fake_images = self.generator(z)#将噪声种子z输入生成器G，产出一批“假图片”
            fake_images = self._ensure_three_channels(fake_images)

            # 训练判别器
            # 我们来看这个过程的分解：
            #
            # errD_fake.backward(): D看完假图片，心里记下了一笔“学习笔记”（计算并累积了假图片的梯度）。
            #
            # errD_real.backward(): D又看完真图片，在刚才的“学习笔记”上，又补充了关于真图片的学习心得（再次计算并累积了真图片的梯度）。
            #
            # self.optimizer_D.step(): D拿出写满了“真假心得”的完整笔记，对自己进行一次全面的总结和提升（根据累积的总梯度，更新自己的权重）。
            self.optimizer_D.zero_grad()
            label = torch.full((batch_size,), fake_label, device=device, dtype=torch.float)
            output = self.discriminator(fake_images.detach())#将G生成的假图片输入判别器D。.detach()是关键，它会阻断梯度从D流向G。因为在这个步骤，我们只想更新D，不想影响G。
            errD_fake = self.criterion_bce(output, label)#errD_fake 是 Loss
            errD_fake.backward()#根据loss计算梯度

            # 使用真实代理样本（从缓存获取）
            if proxy_batches:
                real_data = proxy_batches[i % len(proxy_batches)]
                if real_data.size(0) == batch_size:
                    label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
                    output = self.discriminator(real_data)
                    errD_real = self.criterion_bce(output, label)#计算D在真图片上的损失。D的目标是把这些图片识别为1。
                    errD_real.backward()# 计算这部分损失关于D的梯度。
                    errD = errD_fake + errD_real#将两部分损失相加，得到D的总损失。
                else:
                    errD = errD_fake
            else:
                errD = errD_fake

            self.optimizer_D.step()#根据计算出的总梯度，使用优化器optimizer_D来更新判别器D的权重。

            # 训练生成器
            self.optimizer_G.zero_grad()
            # 对抗损失
            label.fill_(real_label)
            output = self.discriminator(fake_images)#将之前生成的同一批假图片再次输入D。注意，这里没有.detach()，因为我们需要梯度能够从D反向传播回G。output就是鉴定师D打出的一批分数（一个包含很多0到1之间数字的张量），代表了他认为这批画有多“真”。
            adv_loss = self.criterion_bce(output, label)# 计算对抗性损失。这是G的第一个目标：生成能让D判断为“真”的图片。G的目标是什么？它希望自己画的画，能让鉴定师D打出尽可能高的分，最好是满分1分。
            #label被我们手动设置成了real_label（也就是1）。所以，上面行代码计算的是：鉴定师打出的分数(output)和我们的期望分数(1)之间的差距。我们要最小化损失函数


            # 类多样性损失
            # 计算div_loss的流程：
            # G画了一批画（fake_images）。
            # 我们把这些画拿给克隆模型C去看（clone_outputs = self.clone_model(fake_images)）。
            # C会对每幅画给出一个预测，比如“这幅画80 % 像猫，10 % 像狗...”，“那幅画90 % 像飞机，5 % 像卡车...”。
            # class_diversity_loss函数会计算这一整批画的平均预测概率。
            # 举例：如果G只画了“猫”和“狗”，那么平均下来，“猫”和“狗”的平均概率就会很高，而“飞机”、“卡车”等其他类别的平均概率就会很低。这个平均概率分布就会非常不均衡。
            # 数学上，一个不均衡的概率分布，其 **“熵(Entropy)” ** 很低。这个损失函数的目标就是：最大化这个熵（代码里是最小化负熵，效果等价）。

            # div_loss如何起作用？
            # 当熵很低时（分布不均衡），div_loss就会变得很高。
            # 为了减小这个loss，G就必须调整策略，去画一些能让C预测出其他类别（比如“飞机”、“卡车”）的画。
            with torch.no_grad():#这是一个关键设计。我们用克隆模型C来评估生成图片的多样性，但是我们不希望在这一步更新C的权重，也不希望计算关于C的梯度。所以用no_grad包裹起来。
                clone_outputs = self.clone_model(fake_images)#将假图片输入给克隆模型C，得到C对这批图片的预测结果。clone_outputs 的形状通常是 [batch_size, num_classes]相当于是软标签
            div_loss = self.class_diversity_loss(clone_outputs)#计算类别多样性损失。这是G的第二个目标：生成能让C的预测结果尽可能多样化的图片，避免模式崩溃。
            # class_diversity_loss函数最核心的计算步骤。将“克隆模型自己的软标签”（clone_outputs）转换成“这一批次图片在10个类别中的所占比”，分两步走：
            # 第一步：Softmax - 把每一张图片的“原始分数”变成“概率百分比”。
            # 第二步：求平均 - 把所有图片的“概率百分比”综合起来，算出整个批次的“平均占比”。

            # 总生成器损失
            g_loss = adv_loss + self.lambda_div * div_loss#将两个损失加权相加，得到G的总损失。lambda_div是超参数，用来平衡这两个目标的重要性。
            g_loss.backward()#计算总损失关于生成器G的梯度
            self.optimizer_G.step()

            pbar.update(1)
            if (i + 1) % 100 == 0:
                pbar.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{errD.item():.4f}")

        pbar.close()
        print("生成器初始化完成")

    # 源源不断地、自动地提供下一批“代理数据”（Proxy Data）。
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

    #get_hard_label 函数最终返回的就是一个包含了这批图片中，每一张图片分别属于哪一类的索引的张量。
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
        self.clone_model.eval()#将克隆模型C设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:#这里的self.testloader就是我们之前讨论过的那个公正的“考官”——官方的CIFAR-10测试集。这个循环会遍历整个测试集中的所有图片。
                # 这几行是标准模型测试流程：
                #
                # 将一批测试图片inputs和它们的真实标签labels放到GPU上。
                #
                # 用我们的克隆模型clone_model对图片进行预测。
                #
                # predicted.eq(labels): 比较模型的预测和真实标签是否相等。
                #
                # 累加总数total和预测正确的数量correct。
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
            save_path = os.path.join(self.output_dir, 'best_clone_model.pth')
            torch.save(self.clone_model.state_dict(), save_path)
            print(f"保存新的最佳模型至 {save_path}，准确率: {accuracy:.2f}%")

        return accuracy

    # 因此，这个函数的核心目的就是：
    #
    # 快速构建一个“基础版”的克隆模型：利用已经预训练好的生成器（Generator）产生一批数据，然后去查询受害者模型（Victim
    # Model）获得这些数据的标签。
    #
    # 监督训练：用这些“生成数据 - 受害者标签”对，对克隆模型进行一个标准的监督学习训练。
    #
    # 提供一个更好的起点：经过这个“热身”阶段后，克隆模型不再是完全随机的，它已经初步具备了模仿受害者模型的能力。这为后续更精细化的、代价高昂的交替训练提供了一个更好的、更稳定的起点，从而提升整个模型窃取攻击的效率和成功率。
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
            self.optimizer_C.step()#更新参数
            # self.scheduler_C.step()#更新学习率

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


    # class_diversity_loss 函数的作用是计算一个“类多样性损失”，其核心目的是为了驱动生成器（Generator）产生在类别上分布更均匀、更多样化的图片
    def class_diversity_loss(self, batch_outputs):
        """计算类多样性损失 - 严格按照论文公式5"""
        batch_size = batch_outputs.size(0)
        softmax_outputs = torch.softmax(batch_outputs, dim=1)# 对克隆模型的原始输出 batch_outputs 进行 softmax 操作。softmax 会将每一行的输出（代表一张图片的预测）转换成一个概率分布，其中每个元素都在0到1之间，且所有元素的和为1。dim=1 指定了在“类别”这个维度上进行操作。现在 softmax_outputs 的每一行都可以被看作是克隆模型认为这张图片属于10个类别的各自概率。

        # 计算每个类别的平均置信度
        alpha_j = softmax_outputs.mean(dim=0)  # shape: [num_classes]这是最核心的一步。它计算了 softmax_outputs 在批量（batch）这个维度上的平均值 (dim=0)。

        # 计算负熵: Lclass_div = ∑(j=0 to K) [αj log αj]
        # 避免log(0)，添加小epsilon
        loss = torch.sum(alpha_j * torch.log(alpha_j + 1e-10))

        return loss  # 最小化负熵

    #visualize_class_distribution 函数是一个诊断和可视化工具，它的核心作用是检查生成器（Generator）当前生成图像的类别多样性。
    # 这个函数通过以下步骤工作：
    #
    # 让生成器产生大量（例如1000张）的假图片。
    #
    # 将这些假图片全部交给受害者模型进行预测，得到每个图片的标签。
    #
    # 统计这1000个预测标签中，每个类别（0到9）分别出现了多少次。
    #
    # 将统计结果绘制成一个柱状图并保存为图片。
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

    #train 函数实现了DFMS-HL算法的核心——生成器（G）、判别器（D）和克隆模型（C）之间的“交替训练”机制。
    # 它执行以下两个阶段的交替循环：
    #
    # 阶段一：提升数据质量（训练G和D）
    #
    # 在这个阶段，克隆模型C的参数被固定。
    #
    # 生成器G的目标是产生既能骗过判别器D，又能让克隆模型C认为类别多样的图片。
    #
    # 判别器D的目标是尽力分辨真实数据（来自代理数据集）和G生成的假数据。
    #
    # 通过这个博弈过程，G被“逼迫”着去生成质量更高、类别更丰富的查询数据。
    #
    # 阶段二：提升克隆性能（训练C）
    #
    # 在这个阶段，生成器G和判别器D的参数被固定。
    #
    # 我们使用刚刚训练得更好的G来生成一批新的查询数据。
    #
    # 将这些数据发送给受害者模型进行查询，获取标签。
    #
    # 用这些（图片，标签）对来训练克隆模型C，提升其模仿能力。
    #
    # 这两个阶段不断交替，形成了一个正向循环：更好的G能产生更好的数据，更好的数据能训练出更好的C；而一个更好的C又能为G的class_diversity_loss提供更准确的指导信号，从而进一步提升G。这个循环会一直持续，直到耗尽查询预算。
    #
    # 真样本（代理数据）贯穿始终是正确的，它的作用是稳定GAN的训练，为生成器提供一个稳定的“图像真实感”的基准。
    #
    # 生成器并不是盲目地靠近代理数据。它是在“看起来像真实图片”（由代理数据和判别器约束）和“语义上能被受害者模型识别成不同类别”（由克隆模型和类多样性损失约束）这两个目标之间寻找一个绝妙的平衡点。
    def train(self, num_queries=8000000, batch_size=128, g_steps=1, c_steps=1, evaluate_every=100000):#num_queries: 总查询预算，默认为800万次，这是攻击成本的上限。设定的目的: 设定这个预算主要是为了凸显其方法的查询效率。在同一篇论文中，作者反复提到一个作为对比的基线方法 ZSDB3KD，该方法需要高达40亿次（4000 million）的查询才能达到类似效果。 通过将自己的查询预算限制在800万次，作者有力地证明了他们的方法在成本上降低了约500倍，这是一个巨大的进步。
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
            self.discriminator.train()#将G和D设置为“训练模式”。
            self.clone_model.eval()#关键操作。将C设置为“评估模式”。这会固定C的参数，并关闭其Dropout等层，确保它在计算class_diversity_loss时提供一个稳定、确定的输出。

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
                # 使用生成的假样本，训练假样本
                label = torch.full((batch_size,), fake_label, device=device, dtype=torch.float)
                output = self.discriminator(fake_images.detach())
                errD_fake = self.criterion_bce(output, label)
                errD_fake.backward()

                # 使用真实代理样本，训练真样本
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
                #上面是标准的GAN中判别器的训练流程。它分别计算D在假图片和真图片（来自代理数据集）上的损失，然后将总损失反向传播，更新D的参数，让它更擅长分辨真假。

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
            self.generator.eval()#关键操作。将G设置为“评估模式”，固定其参数
            self.clone_model.train()#将C切换回“训练模式”，准备更新它的参数

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
        final_model_path = os.path.join(self.output_dir, 'clone_model_final.pth')
        torch.save(self.clone_model.state_dict(), final_model_path)
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
        history_path = os.path.join(self.output_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
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

        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.tight_layout()
        plt.savefig(plot_path)
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
    VICTIM_MODEL_PATH = "results/0_victim_model/victim_model.pth"
    SYNTHETIC_GEN_PATH = "results/1_gan_training/dcgan_generator_synthetic.pth"
    PROXY_GEN_PATH = "results/1_gan_training/dcgan_generator_proxy.pth"
    OUTPUT_DIR = "results/2_baseline_attack"

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

    # 初始化DFMS_HL
    dfms_hl = DFMS_HL(victim_model, generator, proxy_dataset=proxy_dataset, nc=nc, lambda_div=500,output_dir=OUTPUT_DIR)

    # --- 论文步骤 2: 第一次初始化克隆模型 (C_initial) ---
    print("【步骤1】第一次初始化克隆模型...")
    dfms_hl.initialize_clone(num_iterations=20000, evaluate_every=20000)
    # (请确保这里的学习率调度器问题已经按之前的建议修复)

    # --- 论文步骤 3: 使用 C_initial 微调生成器 ---
    print("【步骤2】微调生成器...")
    dfms_hl.initialize_generator(nG=5000)  # 使用较小的迭代次数

    # --- 【关键补充】论文步骤 4: 从头训练一个新的克隆模型 (C_final_initial) ---
    print("【步骤3 - 新增】重新创建并训练克隆模型...")
    # 重新创建克隆模型和优化器，相当于“从头开始”
    dfms_hl.clone_model = create_clone_model().to(device)
    dfms_hl.optimizer_C = optim.SGD(dfms_hl.clone_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    dfms_hl.scheduler_C = CosineAnnealingLR(dfms_hl.optimizer_C, T_max=200)  # 也重新创建调度器

    # 使用微调后的G，再次调用克隆初始化函数，得到一个好的起点
    dfms_hl.initialize_clone(num_iterations=20000, evaluate_every=20000)

    # --- 论文步骤 5: 开始最终的交替训练 ---
    print("【步骤4】开始最终交替训练...")
    history = dfms_hl.train(
        num_queries=3000000,  # 这里的预算可能需要根据前面的消耗进行调整
        batch_size=128,
        g_steps=1,
        c_steps=1,
        evaluate_every=100000
    )
