import torch
from torch import nn

X = torch.rand(2, 20)
file_name = 'mlpparams.txt'
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, X):
        H1 = self.relu(self.hidden(X))
        output = self.output(H1)
        return output
    
def save_data():
    net = MLP()
    Y = net(X)
    # 保存模型参数
    torch.save(net.state_dict(), file_name)
    print(net)
    return Y

def load_data():
    clone = MLP()
    # 加载模型参数
    clone.load_state_dict(torch.load(file_name, weights_only=True))
    clone.eval()
    print(clone)
    return clone(X)

def test_same_data():
    # 测试保存的模型参数和加载的模型参数是否一致
    Y = save_data()
    Y_clone = load_data()
    print(f'Y==Y_clone:{Y==Y_clone}')