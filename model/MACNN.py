import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SEnet import *
from model.ECAnet import *


class MACNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MACNN, self).__init__()
        self.num_classes = num_classes
        # 宽卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=64, stride=2, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(32)
        # 1多尺度卷积池化1*5
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.max_pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 1多尺度卷积池化1*7
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.max_pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 1多尺度卷积池化1*9
        self.conv2_3 = nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=4)
        self.max_pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # SE通道注意力机制1
        self.se1 = SEBlock(64)
        # 2多尺度卷积池化1*5
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.max_pool3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2多尺度卷积池化1*7
        self.conv3_2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.max_pool3_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2多尺度卷积池化1*9
        self.conv3_3 = nn.Conv2d(64, 128, kernel_size=9, stride=1, padding=4)
        self.max_pool3_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # ECA注意力机制2
        self.se2 = SEBlock(128)
        self.eca = ECANet(128*3)
        # 全局平均池化
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        # 全连接层降维
        self.feature = nn.Sequential(
            nn.Linear(128*3,128),nn.BatchNorm1d(128),nn.ReLU())
        self.classification_layer = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.max_pool1(x))

        x1 = F.relu(self.conv2_1(x))
        x1 = F.relu(self.max_pool2_1(x1))
        x1 = F.relu(self.se1(x1))
        x1 = F.relu(self.conv3_1(x1))
        x1 = F.relu(self.max_pool3_1(x1))
        x1 = F.relu(self.se2(x1))

        x2 = F.relu(self.conv2_2(x))
        x2 = F.relu(self.max_pool2_2(x2))
        x2 = F.relu(self.se1(x2))
        x2 = F.relu(self.conv3_2(x2))
        x2 = F.relu(self.max_pool3_2(x2))
        x2 = F.relu(self.se2(x2))

        x3 = F.relu(self.conv2_3(x))
        x3 = F.relu(self.max_pool2_3(x3))
        x3 = F.relu(self.se1(x3))
        x3 = F.relu(self.conv3_3(x3))
        x3 = F.relu(self.max_pool3_3(x3))
        x3 = F.relu(self.se2(x3))

        x = torch.cat([x1, x2, x3], dim=1)  # 多尺度特征融合
        x = self.eca(x)  # eca通道注意力机制加权
        x = self.global_avg_pooling(x)
        feature = x.view(x.size(0), -1)  # 特征展平
        feature = self.feature(feature)
        x = self.classification_layer(feature)  # 全连接层
        x = F.softmax(x, dim=1)
        return x
