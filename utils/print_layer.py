import torch

def print_layer(net, X_shape):
    """检查模型的每一层输出形状"""
    X = torch.rand(size=X_shape, dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)