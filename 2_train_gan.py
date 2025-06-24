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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义合成数据集生成
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


# 定义CIFAR-100作为代理数据集
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


# 训练DCGAN
def train_dcgan(dataloader, synthetic=False, num_epochs=50, save_path='dcgan_generator.pth'):
    """预训练DCGAN生成器"""
    # 创建生成器和判别器
    nz = 100  # 潜在向量的大小
    nc = 1 if synthetic else 3  # 通道数：1用于灰度合成图像，3用于彩色CIFAR-100

    netG = Generator(nz=nz, nc=nc).to(device)
    netD = Discriminator(nc=nc).to(device)

    # 初始化权重
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # 设置优化器
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, device=device)  # 用于可视化
    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print("开始预训练DCGAN...")

    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(dataloader):
            ############################
            # (1) 更新判别器D: 最大化log(D(x)) + log(1 - D(G(z)))
            ###########################
            # 训练使用真实样本
            netD.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)

            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()

            # 训练使用生成的假样本
            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) 更新生成器G: 最大化log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # 我们想要G生成的假样本被认为是真实的
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(
                    f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        # 保存生成的图像
        if epoch % 10 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                # 显示和保存图像
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                if synthetic:
                    # 对于单通道灰度图像
                    plt.imshow(np.transpose(torchvision.utils.make_grid(
                        fake, normalize=True).cpu(), (1, 2, 0)), cmap='gray')
                else:
                    # 对于彩色图像
                    plt.imshow(np.transpose(torchvision.utils.make_grid(
                        fake, normalize=True).cpu(), (1, 2, 0)))
                plt.savefig(f'dcgan_samples_epoch_{epoch}.png')
                plt.close()

    # 保存生成器
    torch.save(netG.state_dict(), save_path)
    print(f'生成器已保存至 {save_path}')

    return netG, netD


# 主函数
if __name__ == "__main__":
    # 使用合成数据
    synthetic_dataset = SyntheticDataset(size=50000)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=128, shuffle=True, num_workers=2)

    # 使用CIFAR-100子集作为代理数据
    proxy_dataset = get_proxy_dataset(num_classes=40)
    proxy_loader = DataLoader(proxy_dataset, batch_size=128, shuffle=True, num_workers=2)

    # 训练使用合成数据的DCGAN
    train_dcgan(synthetic_loader, synthetic=True, num_epochs=50, save_path='dcgan_generator_synthetic.pth')

    # 训练使用代理数据的DCGAN
    train_dcgan(proxy_loader, synthetic=False, num_epochs=50, save_path='dcgan_generator_proxy.pth')