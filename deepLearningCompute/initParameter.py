import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
# m：模块module
def init_normal(m):
    """参数初始化为均值为0，标准差为0.01的正态分布"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

def init_constant(m):
    """参数初始化为常量"""
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def test_init_parameter():
    # 对所有的块应用统一的初始化方法
    net.apply(init_normal)

def test_init_parameter_separated():
    # 对不同的块应用不同的初始化方法
    net[0].apply(init_xavier)
    net[2].apply(init_constant)
    print(net[0].weight.data)
    print(net[2].weight.data, 'a')

def test_init_parameter_separated_1():
    # 对不同的块应用不同的初始化方法
    net.apply(init_constant)
    net[0].apply(init_xavier)
    print(net[0].weight.data)
    print(net[2].weight.data, 'b')


def shared_parameter():
    # net[2]和net[4]是同一个对象shared，共享相同的参数
    shared = nn.Linear(8, 8)
    net = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(),
        shared, nn.ReLU(),
        shared, nn.ReLU(),
        nn.Linear(8, 1)
    )
    X = torch.rand(2, 8)
    net(X)