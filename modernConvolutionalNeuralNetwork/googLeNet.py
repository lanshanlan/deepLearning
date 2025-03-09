import torch
from torch import nn
from utils.loadData import load_data_fashion_mnist
from utils.train_ch6 import train_ch6
from utils.print_layer import print_layer
from utils.gpu import try_gpu
from torch.nn import functional as F

# GoogLeNet卷积神经网络
# ImageNet挑战赛2014年第一名的模型
# 设计了一个Inception块,其由多条并行路径组成
class Inception(nn.Module):
    # c1--c4是没跳路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 路径1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 路径2,1x1卷积层后接上3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 路径3,1x1卷积层后接上5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 路径4，3x3最大汇聚层后接上1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连接输出
        return torch.cat((p1, p2, p3, p4), dim=1)

# 下面逐一实现GoogLeNet的5个模块
# 模块一输出通道为64,使用一个7x7卷积层
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 模块二使用两个卷积层，输出通道为192
# 第一个为1x1卷积层，第二个为3x3卷积层
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 模块三串联2个Inception块,加上一个3x3最大汇聚层，输出通道为480
b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 模块四串联5个Inception块，加上一个3x3最大汇聚层，输出通道为832
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 模块五串联2个Inception块，加上一个全局平均汇聚层
# 最后将输出展平为二维数组，再连接一个输出个数为标签类别数的全连接层
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

def print_googLeNet_layer():
    X_shape = (1, 1, 96, 96)
    print_layer(net, X_shape)

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

def test_googLeNet():
    lr, num_epochs = 0.1, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())