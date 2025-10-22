import torch.nn as nn
import torch
import math

class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):         #in_channels表示输入的通道数，gamma和b是可选参数，分别用于计算卷积核大小的调整。
        super(ECANet, self).__init__()   #初始化函数
        self.in_channels = in_channels    #将输入通道数保存在模型中的in_channels属性中
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))       #定义了一个自适应平均池化层，用于对输入特征图进行池化操作，将其池化为大小为(1, 1)的特征图。
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))    #计算卷积核的大小。通过一个公式计算，使用输入通道数in_channels、gamma和b来确定卷积核的大小
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1   #确保卷积核的大小为奇数，因为一般情况下卷积核的大小是奇数。
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)     #输入通道数为1，输出通道数为1，卷积核的大小为kernel_size，填充的大小使得卷积操作后特征图的大小不变，bias参数设置为False表示不使用偏置项。
        self.act1 = nn.Sigmoid()  #对卷积层的输出进行激活，将其限制在0到1之间。

    def forward(self, x):   #定义了前向传播
        output = self.fgp(x)    #将输入张量x通过自适应平均池化层self.fgp进行池化操作，得到池化后的输出特征图output。
        output = output.squeeze(-1).transpose(-1, -2)   #首先使用squeeze(-1)函数去除维度为1的维度，然后使用transpose(-1, -2)函数交换维度，这一步通常用于将特征图的通道维度放在最后。
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)    #将处理后的特征图output通过一维卷积层self.con1进行卷积操作，然后再次使用transpose(-1, -2)交换维度，最后使用unsqueeze(-1)在最后添加一个维度，这一步通常用于将通道维度放在最后
        output = self.act1(output)   #将卷积后的特征图通过Sigmoid激活函数self.act1进行激活，将其值限制在0到1之间。
        output = torch.multiply(x, output) #将输入张量x与激活后的特征图output进行逐元素相乘操作，得到最终的输出特征图。
        return output  #回最终的输出特征图
