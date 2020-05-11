import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

np.random.seed(0)
torch.manual_seed(0)

# transformer
# SNE -> TNE
from torch.nn.modules.transformer import Transformer

trans = Transformer(d_model=10, nhead=2)
src = torch.randn(20, 32, 10)
tgt = torch.randn(10, 32, 10)
assert trans(src, tgt).size() == (10, 32, 10)

# Linear 矩阵相乘
m = nn.Linear(in_features=20, out_features=30)
n = torch.randn(50, 20)
assert m(n).size() == (50, 30)

# bmm
m = torch.randn(10, 3, 4)
n = torch.randn(10, 4, 6)
assert torch.bmm(m, n).size() == (10, 3, 6)

# lstm
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
input1 = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)

output, (h0, c0) = lstm(input1, (h0, c0))

#  embedding
m = nn.Embedding(num_embeddings=10, embedding_dim=3)
# 常见错误  Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.FloatTensor
n = torch.LongTensor([[1, 3, 4, 5], [2, 3, 6, 7]])
print(n.type())
assert m(n).size() == (2, 4, 3)

# squeeze
x = torch.randn(3, 1, 4, 1)
print(x.size())
print(x.squeeze().size())

# unsqueeze
x = torch.randn(1, 4, 5)
print(x.size())
print(x.unsqueeze(2).size())
print(x.unsqueeze(1).size())

# view
x = torch.randn(4, 4)
print(x.view(-1, 8).size())
print(x.view(-1, 8).is_contiguous())
print(x.view(-1, 8).contiguous())

# conv2d
m = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=3, stride=2)
input1 = torch.randn(20, 16, 50, 100)
print(m(input1).size())

# transpose
x = torch.randn(3, 4)
print(x.size())
print(x.transpose(1, 0).size())

#  permute
x = torch.randn(3, 4, 5, 6)
print(x.size())
print(x.permute(3, 2, 1, 0).size())
