import torch
import torchvision.models as models
import torch.nn as nn


class ResNet(nn.Module):          #定义ResNet
    def __init__(self, num_class=10):       #它接受一个参数num_class，默认值为2，用于指定分类任务的类别数量。
        super(ResNet, self).__init__()        #初始化参数

        self.num_class = num_class         #表示分类任务的类别数量。
        self.drou = nn.Dropout(0.2)     #训练过程中随机将输入张量的部分元素置零，防止过拟合。Dropout的丢弃概率为0.2

        resnet50 = models.resnet50()      #预训练的ResNet-50模型
        for p in resnet50.parameters():      #设置ResNet-50模型的所有参数都需要梯度计算，以便在训练过程中更新参数。
            p.requires_grad = True

        self.features = nn.Sequential(*list(resnet50.children())[:-1])  #定义了一个特征提取部分，即ResNet-50模型的所有层，除了最后一层全局平均池化层

        self.classifier = nn.Linear(128, self.num_class)    #定义了一个全连接层，将提取的特征映射到类别数量的输出空间。输入特征维度为128，输出特征维度为类别数量
        self.feature = nn.Sequential(
            nn.Linear(2048, 128), nn.ReLU())    #通过一个线性层将ResNet-50的最后一个特征图映射到128维空间，然后通过ReLU激活函数进行非线性变换。

        self._init_weights()       #初始化模型参数的权重

    def forward(self, x): #定义了前向传播
        x = self.features(x)     #将输入x通过特征提取部分提取特征
        x = torch.flatten(x, 1)   #将提取的特征展平为一维
        feature = self.feature(x)   #将展平后的特征通过特征提取部分映射到128维特征空间
        x = self.classifier(feature)   #将128维特征通过全连接层映射到类别数量的输出空间
        x = self.drou(x)   #对全连接层的输出进行Dropout操作，以防止过拟合
        #x = torch.log_softmax(x, dim=1)  #对全连接层的输出进行log_softmax操作，得到类别预测的对数概率。
        return feature, x   #返回128维特征和类别预测的对数概率作为模型的输出

    def _init_weights(self):       #初始化模型参数的权重。
        for m in self.modules():    #遍历模型的所有模块
            if isinstance(m, nn.Linear):      #如果当前模块是线性层
                nn.init.kaiming_normal_(m.weight)     #使用Kaiming正态分布初始化线性层的权重。
                m.bias.data.normal_(0.0, 0.001)         #将线性层的偏置项初始化为均值为0，标准差为0.001的正态分布。


# if __name__ == '__main__':
#     input_test = torch.ones(10, 3, 128, 128)
#     model = ResNet()
#     y, z = model(input_test)
#     print(y.shape, z.shape)
