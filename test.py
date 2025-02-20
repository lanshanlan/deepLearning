import torch
import numpy
import time
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):
    """y=xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

# true_w = torch.tensor([2,-3.4])
# true_b = 4.2
# features, labels = synthetic_data(true_w, true_b, 10)
# print(f'features={features}, labels={labels}')
a = torch.tensor([[1,2],
                  [3,4]])
b = torch.tensor([[1,1],
                  [1,1]])
print(torch.matmul(a,b))