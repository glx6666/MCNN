import argparse
import os
import numpy as np
import math
import sys
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

os.makedirs("images", exist_ok=True)     #创建一个名为 “images” 的文件夹，如果该文件夹已经存在则忽略，参数 exist_ok=True 表示存在时不抛出异常
# 模型文件配置参数
parser = argparse.ArgumentParser()   #创建一个参数解析器
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
opt = parser.parse_args()  #解析并获取用户输入的参数，并将其保存在 opt 变量中
print(opt)     #打印用户输入的参数，便于在训练过程中进行调试和监控

#  图像尺寸
img_shape = (opt.channels, opt.img_size, opt.img_size)  #根据用户输入的参数，构建图像的形状信息，包括通道数、图像高度和宽度。

# 在cuda上运行
cuda = True if torch.cuda.is_available() else False      #检查系统是否有可用的 CUDA，如果有则将 cuda 设置为 True，表示可以在 GPU 上运行模型，否则设置为 False，将在 CPU 上运行模型。


class Generator(torch.nn.Module):              #定义一个网络模型
    def __init__(self, channels):                  #接受一个参数 channels，表示图像的通道数。
        super().__init__()             #始化父类的属性。
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(             #使用 nn.Sequential 封装。这些层将逐步将输入向量转换为输出图像
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            #输入通道数为 100，对应于输入的随机向量、 out_channels=1024：输出通道数为 1024，即输出特征图的深度、kernel_size=4：卷积核大小为 4x4、stride=1：步幅为 1、padding=0：填充为 0。
            nn.BatchNorm2d(num_features=1024),     #num_features=1024 表示归一化的特征数。
            nn.ReLU(True),      #ReLU 激活函数

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=16, out_channels=channels, kernel_size=4, stride=2, padding=1))
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()     #最后一层使用 Tanh 激活函数将输出限制在 [-1, 1] 的范围内，以生成适合图像的像素值

    def forward(self, x):         #定义了数据在模型中的正向传播过程
        x = self.main_module(x)        #x经过main_module变换生成图像
        return self.output(x)          #返回图像


class Discriminator(torch.nn.Module):               #定义一个网络模型
    def __init__(self, channels):         #接受一个参数 channels，表示图像的通道数
        super().__init__()        #初始化父类属性
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            #输入通道数、输出通道256、卷积核大小4*4、步长2、填充大小为1
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),      #256表示归一化的通道数，即输入特征图的深度。在这里，我们对256个特征图进行实例归一化。当设置为True时，模型可以学习归一化的缩放和平移参数。
            nn.LeakyReLU(0.2, inplace=True),         #泄漏系数（leaky coefficient），指定为0.2。它表示当输入为负值时，泄漏 ReLU 允许小的梯度通过，而不是完全置零。
                                                                  #当设置为True时，将会在原始张量上进行操作，节省内存消耗和运算时间。
            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):          #定义一个前向传播
        x = self.main_module(x)       #它将输入x传递给main_module中的各个层，依次进行计算和转换，最终得到一个输出张量。
        return self.output(x)          #将经过main_module处理后的输出张量x传递给self.output，这里假设self.output是网络的输出层

    def feature_extraction(self, x):         #特征提取
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)              #输入张量 x 通过 main_module 中的各个层，依次进行计算和转换，最终得到一个特征张量。

        return x.view(-1, 1024 * 4 * 4)        #通过 view 方法对提取的特征张量 x 进行形状变换。在这里，view 方法被用于将特征张量展平成一个二维张量，第一个维度为 -1 表示自动推断，
                                               # 第二个维度计算为 1024 * 4 * 4，即假设提取的特征张量是 1024 个通道，
                                               # 每个通道的尺寸为 4x4。这样，最终得到一个形状为 (batch_size, 1024 * 4 * 4) 的二维张量，其中 batch_size 是输入张量 x 的批量大小。

# 自定义的数据集加载器，为了让自己的数据集能够适配模型的输入
class CustomDataset(Dataset):       #定义数据集类
    def __init__(self, root_dir, transform=None):       #初始化数据集对象。它接受两个参数：root_dir 表示数据集所在的根目录，transform 是一个可选参数，表示对图像进行的预处理操作。
        self.root_dir = root_dir
        self.transform = transform  # 图像预处理
        self.img_names = [name for name in os.listdir(root_dir) if name.endswith('.png')]   #获取数据集中所有以 .png 结尾的图像文件名，并将它们存储在 self.img_names 列表中。

    def __len__(self):       #返回数据集的长度
        return len(self.img_names)  # 返回图像长度，表示数据集中包含的图像数量。

    def __getitem__(self, idx):
        name = self.img_names[idx]      #根据给定的索引 idx 获取对应位置的图像文件名
        img_name = os.path.join(self.root_dir, self.img_names[idx])       #构造了图像文件的完整路径，通过将根目录 self.root_dir 和图像文件名 self.img_names[idx] 进行连接
        image = Image.open(img_name)   #打开图像文件，得到一个图像对象 image。
        # image = Image.open(img_name)
        image = image.convert('L')
        #image = self.transform(image)
        #是否进行图像预处理
        if self.transform:      #判断是否有预处理操作
            image = self.transform(image)          #对图像对象 image 进行预处理，预处理操作通过 self.transform 实现。

        return image, name              #返回处理后的图像对象 image 和对应的图像文件名 name


# 定义数据转换
transform = transforms.Compose([                      #图像预处理操作的管道，可以包含多个预处理操作
    transforms.Resize(opt.img_size),                  #添加了一个图像尺寸调整的预处理操作，opt.img_size表示调整后的图像尺寸
    transforms.Grayscale(num_output_channels=3),      #添加了一个将彩色图像转换为灰度图像的预处理操作，使用了 transforms.Grayscale 方法。
                                                      # num_output_channels=3 参数指定了输出的通道数为 3，即将彩色图像转换为灰度图像后，每个像素的值都会在三个通道上保持一致。
    transforms.ToTensor(),                            #将图像转换为张量的预处理操作
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])   #对张量进行标准化的预处理，mean 和 std 参数分别指定了每个通道的均值和标准差
])

# 创建 CustomDataset 数据集
custom_dataset = CustomDataset(
    root_dir=r"D:/MACNN-Master3/MACNN-Master/dataset/GAF/CWRU_0/0",
    transform=transform)

# 创建 DataLoader
dataloader = DataLoader(custom_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)   #DataLoader创建数据加载器对象。
                                      #加载的数据集对象、批次、随机打乱顺序、用于数据加载的子进程数量
# Loss weight for gradient penalty
lambda_gp = 10         #损失梯度惩罚系数

# Initialize generator and discriminator
generator = Generator(opt.channels)          #创建了一个生成器对象
discriminator = Discriminator(opt.channels)  #创建一个判别器对象

if cuda:    #如果CUDA可用
    generator.cuda()         #将生成器移动到GPU上执行计算。
    discriminator.cuda()      #将判别器对象也移动到GPU上执行计算

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))   #generator.parameters() 返回生成器中所有需要优化的参数，
                                              # 这些参数将会在训练过程中通过梯度下降进行更新。lr=opt.lr 设置了学习率，betas=(opt.b1, opt.b2) 则设置了 Adam 算法中的两个衰减因子。
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor     #如果CUDA可用（即 cuda 变量为 True），则 Tensor 被定义为 torch.cuda.FloatTensor，
                                           # 表示在GPU上进行计算时所用的张量数据类型。如果CUDA不可用，则 Tensor 被定义为 torch.FloatTensor，表示在CPU上进行计算时所用的张量数据类型。
def calculate_gradient_penalty(D, real_images, fake_images, cuda):       #计算梯度惩罚
    eta = torch.FloatTensor(opt.batch_size, 1, 1, 1).uniform_(0, 1)     #创建了一个大小为 opt.batch_size 的随机张量 eta，其取值范围在 0 到 1 之间，用于插值计算。
    eta = eta.expand(opt.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))  #将 eta 张量沿着维度扩展，以匹配真实图像的大小。这是为了后续的插值计算
    if cuda:          #如果CUDA可用
        eta = eta.cuda()      #将 eta 张量移动到GPU上
    else:
        eta = eta       #保持在CPU上

    interpolated = eta * real_images + ((1 - eta) * fake_images)  #使用线性插值法创建混合图像 interpolated，其中 eta 作为插值因子，用于在真实图像 real_images 和生成图像 fake_images 之间进行插值。

    if cuda:          # 如果CUDA可用
        interpolated = interpolated.cuda()      #将插值后的图像 interpolated 移动到GPU上
    else:
        interpolated = interpolated   #保持在CPU上

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)  #将插值后的图像 interpolated 包装成 PyTorch 的变量，并设置 requires_grad=True，以便后续可以计算其梯度

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)          #用判别器 D 对插值后的图像 interpolated 进行判别，得到其概率。

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(                    #torch.ones(prob_interpolated.size())创建了一个与 prob_interpolated 张量具有相同形状的张量，并且其中的所有元素都被设置为 1
                                  prob_interpolated.size()).cuda() if cuda else torch.ones(
                                  prob_interpolated.size()),
                              create_graph=True, retain_graph=True)[0]   #利用自动求导模块 autograd.grad 计算了判别器对插值图像的概率 prob_interpolated 关于插值图像 interpolated 的梯度。

    # flatten the gradients to it calculates norm batchwise
    gradients = gradients.view(gradients.size(0), -1)          #将梯度张量展平，以便在每个批次上计算梯度的范数。

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()     #计算梯度惩罚，即梯度的范数与 1 之间的差异的平方，并对所有样本取平均。
    return grad_penalty    #返回梯度惩罚


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):         #控制训练总数
    for i, (imgs, names) in enumerate(dataloader):    #遍历数据加载器中的每个批次。imgs 是一个批次的真实图像数据，names 是对应图像的文件名
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))       #将输入的真实图像转换为可求导的变量，并根据设备的可用性将其移动到 GPU 上（如果可用）

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()  #零鉴别器的梯度

        # Sample noise as generator input
        z = Variable(torch.rand((opt.batch_size, 100, 1, 1))).cuda() #生成随机噪声 z，其形状为 (批大小, 100, 1, 1)，然后将其转换为可求导的变量，并将其移动到 GPU 上

        # z = torch.rand((self.batch_size, 100, 1, 1))

        # Generate a batch of images
        fake_imgs = generator(z)    #使用生成器模型生成一批假图像

        # Real images
        real_validity = discriminator(real_imgs)      #将真实图像输入鉴别器，计算其真实性得分
        # Fake images
        fake_validity = discriminator(fake_imgs)    #生成器生成的假图像输入鉴别器，计算其真实性得分。
        # Gradient penalty
        gradient_penalty = calculate_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, cuda)  #计算梯度惩罚，于提高鉴别器的训练稳定性。
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty   #计算鉴别器的损失，其中包括了真实图像的负均值、假图像的正均值以及梯度惩罚的权重项。

        d_loss.backward()    #反向传播计算鉴别器的梯度
        optimizer_D.step() #根据计算的梯度更新鉴别器的参数

        optimizer_G.zero_grad() #清零生成器的梯度

        # 添加一个变量来标识是否加载预训练权重
        load_pretrained_weights = False  # 可以根据需要设置为 True 或 False

        # 如果加载预训练权重，则加载
        # if load_pretrained_weights:
            # generator.load_state_dict(torch.load("generator_weights_latest.pth"))

        # Train the generator every n_critic steps 每n_critic步训练一次生成器
        if i % opt.n_critic == 0:        #每n_critic训练一次生成器

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)    #生成一些假图片
            # Loss measures generator's ability to fool the discriminator
            # 在生成的假图像中进行训练
            fake_validity = discriminator(fake_imgs)    #假图片输入到鉴别器中计算真实性得分
            g_loss = -torch.mean(fake_validity)        #计算生成器损失，负号表示最小化这个损失

            g_loss.backward()               #生成器反向传播
            optimizer_G.step()                 #更新生成器模型的参数

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if opt.n_epochs - epoch <= 3:         #在最后训练的三个时期中
                for fake, name in zip(fake_imgs, names):        #遍历生成的假图像 fake_imgs 和它们对应的名称 names
                    name = name.split(".")[0]                      #将每个图像的名称 name 按照文件扩展名分割开，然后选择分割结果的第一个部分作为新的 name
                    save_image(fake,
                               f"D:/MACNN-Master3/MACNN-Master/dataset/wgan_gaf/CWRU_0/0/{name}_{epoch}.png",
                               nrow=1, normalize=True)                       #nrow=1 表示将所有图像排列成一行、normalize=True 则表示将图像像素值标准化至 0 到 1 之间
            #保存生成器的权重
            if batches_done % opt.sample_interval == 0:
                torch.save(generator.state_dict(), "generator_weights_latest.pth")

            batches_done += 1
