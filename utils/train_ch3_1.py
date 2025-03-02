import torch
from matplotlib import pyplot as plt
from utils.activatefunc import accuracy, sgd, cross_entropy, softmax
from IPython import display
from utils.plotfunc import set_axes, set_figsize, use_svg_display
from torch import nn
from utils.loadData import load_data_fashion_mnist, show_images, get_fashion_mnist_labels
from utils.train_ch3 import train_ch3

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_epochs = 10
# pytorch不会隐式地调整输入的形状
# 因此我们在线性层前定义了展平层（flatten）来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

def testTrain_ch3_1():
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)