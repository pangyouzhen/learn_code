from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch import optim

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

a = torch.Tensor(
    [
        [1, 5, 5, 2],
        [9, -6, 2, 8],
        [-3, 7, -9, 10]
    ]
)
print(torch.argmax(a))
print(torch.argmax(a, dim=0))
print(torch.argmax(a, dim=1))

# element wise
m = torch.Tensor([[1, 2, 3], [4, 5, 6]])
n = torch.Tensor([[4, 5, 6], [1, 2, 3]])
print(torch.mul(m, n))

# dropout 主要解决过拟合的问题
m = nn.Dropout(p=0.2)
inputs = torch.randn(20, 16)
print(m(inputs).size())

# layerNorm 主要加快模型的训练速度，对原始的输入维度没有影响
inputs = torch.randn(20, 5, 10, 10)
m = nn.LayerNorm(10)
print(m(inputs).size())
# layerNorm 是怎么样起作用的，归一化处理？

# mask fill
a = torch.tensor([4, 5, 6, 7])
masked = torch.tensor([1, 1, 0, 0]).bool()
b = a.masked_fill(mask=masked, value=torch.tensor(0))
print(b)

# torch 中的两种相乘方式
# torch.mul() 矩阵点乘，也就是element wise 对应位相乘
# torch.matmul() 矩阵相乘，对应 python 中 x @ y ,torch.mm 只适用于二维

#  pytorch 模型的训练步骤
# 1. 数据 2. 模型 3. 损失函数 4. 优化和拟合


#  判断数据是float 还是long
import numpy as np

x = np.random.randint(10, size=(2, 3))
x = torch.from_numpy(x).long()
print(x.type())

# softmax
x = torch.rand(2, 5).int().float()
print(F.softmax(x, dim=0))
print(F.softmax(x, dim=1).sum())

print(F.softmax(x, dim=1))
print(F.softmax(x, dim=1).sum())

print(torch.sigmoid(x))

#  cat
a = torch.randn(3, 4, 5)
b = torch.randn(4, 4, 5)
assert torch.cat((a, a, a), dim=1).size() == (3, 12, 5)
assert torch.cat((a, a, a), dim=0).size() == (9, 4, 5)
assert torch.cat((a, b), dim=0).size() == (7, 4, 5)


# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2), )

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2), )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, *input: Any, **kwargs: Any):
        x = kwargs.get("x")
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)
trainloader = torch.utils.data.Dataloader()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
print("finish")
