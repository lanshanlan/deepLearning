import torch
from torch import nn

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # module是Module子类的一个实例，我们把它保存在Module类的成员
            #_modules中，_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历他们
        for block in self._modules.values():
            X = block(X)
        return X
    
def testMySequential():
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    X = torch.rand(2, 4)
    y = net(X)
    print(X)
    print(y)

# block块
def block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net

rgnet = nn.Sequential(
    block2(),
    nn.Linear(4, 1)
)

def test_block():
    X = torch.rand(2, 4)
    # 查看嵌套块rgnet
    print(rgnet)
    print(rgnet(X))
    # 查看第一个主要块中第二个子块的第一层的偏置bias
    print(rgnet[0][1][0].bias.data)