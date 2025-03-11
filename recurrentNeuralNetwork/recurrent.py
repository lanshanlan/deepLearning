import torch
from torch import nn
from utils.plotfunc import plot
from utils.loadData import load_array
from utils.train_ch3 import evaluate_loss

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x_sin = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

def test_plot_sin():
    plot(time, [x_sin], 'time', 'x', xlim=[1, 1000], figsize=[6, 3])

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x_sin[i: T - tau + i]
labels = x_sin[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = load_array((features[:n_train], labels[:n_train]),
                        batch_size, is_train=True)
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机MLP
def get_net():
    net = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    net.apply(init_weights)
    return net

net = get_net()
# 平方误差注意：MSELoss计算平方误差不带系数1/2
loss = nn.MSELoss(reduction='none')
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {evaluate_loss(net, train_iter, loss):f}')
        
def test_markov_loss():
    train(net, train_iter, loss, 5, 0.01)

def test_onestep_pred():
    test_markov_loss()
    onestep_preds = net(features)
    print(x_sin.shape, onestep_preds.shape)
    plot([time, time[tau:]],
         [x_sin.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

# 测试了一下在已知的batch_size+tau=604步的数据后，预测更后面的输出，结果发现预测结果很快衰减成了一个常数
# 这是因为误差的累积。每次预测的结果用于下一次的预测，每一步的误差累积起来导致误差急剧增大
def test_multistep_pred():
    test_markov_loss()
    onestep_preds = net(features)
    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x_sin[: n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1))
        )
    print(x_sin.shape, onestep_preds.shape, multistep_preds.shape)
    plot([time, time[tau:], time[:]],
         [x_sin.detach().numpy(),
          onestep_preds.detach().numpy(),
          multistep_preds[:].detach().numpy()
          ], 'time',
          'x', legend=['data', '1-step preds', 'multistep preds'],
          xlim=[1, 1000], figsize=(6, 3))

# k步预测，在已知的直到时间步t的数据，预测之后在时间步t+k处的输出，称为k步预测
# 这里测试了1,4,16,64步预测，可以看到1步和4步预测还可以，超过4步后误差急剧增大
def test_more_multistep_pred():
    test_markov_loss()
    max_steps = 64
    features = torch.zeros(T - tau - max_steps + 1, tau + max_steps)
    # 列i(i < tau)来自x的观测，其时间步从（i+1）到（i+T-tau-max_steps+1）
    for i in range(tau):
        features[:, i] = x_sin[i: i + T - tau - max_steps + 1]
    # 列i(i > tau)是来自(i-tau+1)步的预测，其时间步从（i+1）到（i+T-tau-max_steps+1）
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau : i]).reshape(-1)
    steps = (1, 4, 16, 64)
    plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps],
         'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6, 3))