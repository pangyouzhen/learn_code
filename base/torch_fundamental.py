from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch import optim

np.random.seed(0)
torch.manual_seed(0)

# TODO
#  输入：input: batch_size,seq_length
#  embedding:
#       定义： nn.Embedding(vocab_size, embedding_dim)
#       输出: batch_size, seq_length, embedding_dim
#  encoding:
#       LSTM
#           定义: nn.LSTM(embedding_dim, hidden_dim ,num_layers=1, bidirectional=*)
#               forward 可以随机初始化 h0，c0 (num_layers,seq_length,hidden_dim)
#           输出：lstm: batch_size, seq_length, hidden_size 或者是 2*hidden_size
#       transformer:
#       cnn


# transformer
# SNE -> TNE
# pytorch中文档：
# S：src_seq_length
# T: target_seq_length
# N: batch_size
# E: embedding_dim
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
# input: (batch_size, seq_len, embeding_dim)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (h0, c0) = lstm(input1, (h0, c0))
assert output.size() == (5, 3, 20)
output2, (h1, c1) = lstm(input1)
assert output2.size() == (5, 3, 20)

bilstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True)
input1 = torch.randn(5, 3, 10)
output2, (h1, c1) = bilstm(input1)
# bidirectional (batch_size,seq_num,2 * hidden_size)
assert output2.size() == (5, 3, 40)

#  embedding, embedding就是lookup，寻找
# input: (*) -> output: (*,E)
# Embedding的两个参数从 多大的数据中查找，查找的每个字或词的维度是多少
m = nn.Embedding(num_embeddings=10, embedding_dim=3)
# 常见错误  Expected tensor for argument #1 'indices' to have scalar type Long; but got learn_torch.FloatTensor
n = torch.LongTensor([[1, 3, 4, 5], [2, 3, 6, 7]])
print(n.type())
assert m(n).size() == (2, 4, 3)

# TODO 增加padding_idx的作用是什么？
m2 = nn.Embedding(10, 3, padding_idx=0)
input = torch.LongTensor([[0, 2, 0, 5]])
assert m2(input).size() == (1, 4, 3)

weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
input = torch.LongTensor([1])
print(embedding(input))
# assert embedding(input) == torch.Tensor([[4, 5.1, 6.3]])

weight = torch.FloatTensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
                            [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]])
embedding = nn.Embedding.from_pretrained(weight)
n = torch.LongTensor([[1, 3, 4, 5], [2, 3, 6, 7]])
print(embedding(n))

# squeeze
x = torch.randn(3, 1, 4, 1)
print(x.size())
print(x.squeeze().size())

# unsqueeze
x = torch.randn(1, 4, 5)
print(x.size())
print(x.unsqueeze(2).size())
print(x.unsqueeze(1).size())
#  squeeze 挤压，就是删减维度
#  unsqueeze - squeeze的反义词，增加维度

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
print(torch.mul(m, n).size())

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

# learn_torch 中的两种相乘方式
# learn_torch.mul() 矩阵点乘，也就是element wise 对应位相乘
# learn_torch.matmul() 矩阵相乘，对应 python 中 x @ y ,learn_torch.mm 只适用于二维

#  pytorch 模型的训练步骤
# 1. 数据 2. 模型 3. 损失函数 4. 优化和拟合


#  判断数据是float 还是long
import numpy as np

x = np.random.randint(10, size=(2, 3))
x = torch.from_numpy(x).long()
print(x.type())

# softmax
x = torch.rand(2, 5).int().float()
print(F.softmax(x))
print(F.softmax(x, dim=0))
print(F.softmax(x, dim=1).sum())

print(F.softmax(x, dim=1))
print(F.softmax(x, dim=1).sum())

print("激活函数-----")
print(F.tanh(x).size())

print(torch.sigmoid(x).size())

#  cat
#  cat 除了指定的轴维度不同，其他的必须相同
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
# 这一段代码有问题
# trainloader = learn_torch.utils.data.dataloader
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(cnn.parameters(), lr=0.01)
# for epoch in range(2):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = cnn(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
# print("finish")

# (batch_size,in_channels,H_in,W_in) -> (batch_size,out_channels,H_out,W_out)
m = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=3, stride=2)
# m2 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# m3 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
# H_out = (50 + 2 * 0 - 0 *(3-1) -1) / 2+ 1
# W_out =
print(m(input).shape)
# learn_torch.Size([20, 33, 24, 49])

# print(m2(input).shape)
# print(m3(input).shape)

# 损失函数-交叉熵
# Input: (minibatch,C) where C = number of classe
# target: (minibatch)


x_input = torch.randn(3, 3)  # 随机生成输入
print('x_input:\n', x_input)
y_target = torch.tensor([1, 2, 0])  # 设置输出具体值 print('y_target\n',y_target)

# 计算输入softmax，此时可以看到每一行加到一起结果都是1
softmax_func = nn.Softmax(dim=1)
soft_output = softmax_func(x_input)
print('soft_output:\n', soft_output)

# 在softmax的基础上取log
log_output = torch.log(soft_output)
print('log_output:\n', log_output)

# 对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
logsoftmax_func = nn.LogSoftmax(dim=1)
logsoftmax_output = logsoftmax_func(x_input)
print('logsoftmax_output:\n', logsoftmax_output)

# pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
nllloss_func = nn.NLLLoss()
nlloss_output = nllloss_func(logsoftmax_output, y_target)
print('nlloss_output:\n', nlloss_output)

# 直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
crossentropyloss = nn.CrossEntropyLoss()
crossentropyloss_output = crossentropyloss(x_input, y_target)
print('crossentropyloss_output:\n', crossentropyloss_output)

# x_input:
#  tensor([[-0.5683,  0.6400, -1.5926],
#         [ 0.9784, -0.3355, -0.5546],
#         [-0.1254, -1.1503, -0.2112]])
# soft_output:
#  tensor([[0.2125, 0.7113, 0.0763],
#         [0.6736, 0.1810, 0.1454],
#         [0.4392, 0.1576, 0.4031]])
# log_output:
#  tensor([[-1.5490, -0.3407, -2.5733],
#         [-0.3952, -1.7091, -1.9282],
#         [-0.8227, -1.8476, -0.9085]])
# logsoftmax_output:
#  tensor([[-1.5490, -0.3407, -2.5733],
#         [-0.3952, -1.7091, -1.9282],
#         [-0.8227, -1.8476, -0.9085]])
# y_target = torch.tensor([1, 2, 0])
# nlloss_output:
#  (0.3407 + 1.9282 + 0.8227) /3
#  tensor(1.0305)
# crossentropyloss_output:
#  tensor(1.0305)


# 这个为什么会报 target out of bounds, NllLoss = -x[class] 的期望，x[class] 超出索引
# x_input = torch.randn(3, 2)
# y_target = torch.tensor([1, 2, 0])
# crossentropyloss = nn.CrossEntropyLoss()
# crossentropyloss_output = crossentropyloss(x_input, y_target)

score = torch.randn(3, 2)
label = torch.Tensor([1, 0, 1]).long()
crossentropyloss_output2 = crossentropyloss(score, label)

m1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
m2 = torch.nn.MaxPool1d(kernel_size=3, stride=2)
inputm = torch.randn(2, 4, 5)
print(m1(inputm).shape)
print(m2(inputm).shape)

m1 = torch.nn.AvgPool2d(kernel_size=3, stride=2)
m11 = torch.nn.AvgPool2d(kernel_size=3)
m2 = torch.nn.AvgPool1d(kernel_size=3, stride=2)
inputm = torch.randn(2, 4, 5)
print(m1(inputm).shape)
print(m2(inputm).shape)
print(m11(inputm).shape)

target = torch.randn(12, 5)
out = torch.randn(12, 5)
loss = -target * torch.log(out) - (1 - target) * torch.log(1 - out)
print(loss)
loss2 = loss.sum(-1).mean()
print(loss2)
# crition = torch.nn.BCELoss()
# print(crition(target, out) == loss)

tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
print(torch.matmul(tensor1, tensor2).size())

m = nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3)
# input： N,in_channels,L_in
# 默认情况下： Conv1d：N,out_channels,L_in - kernel_size + 1
# 公式：https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html?highlight=conv1d#torch.nn.Conv1d
input = torch.randn(20, 16, 50)
output = m(input)
print(output.shape)

# 神经网络的初始化方法，
# 正太分布 初始化
w = torch.empty(3, 5)
nn.init.xavier_normal_(w, gain=nn.init.calculate_gain("relu"))
#  均匀分布 初始化
# nn.init.xavier_uniform_(w,)
#  初始化为常数
# nn.init.constant_(w)
# 多头注意力机制
from torch.nn.modules.activation import MultiheadAttention

query = torch.randn(11, 20, 40)
key = torch.randn(6, 20, 40)
value = torch.randn(6, 20, 40)
attn = MultiheadAttention(embed_dim=40, num_heads=4)
print(attn)
for param in attn.named_parameters():
    # print(param,param.size())
    print(param[0], "++", param[1].shape)

# pytorch 归一化层：BatchNorm、LayerNorm、InstanceNorm、GroupNorm
# Norm 最归一化，所以输入输出的维度是不变化的
# BatchNorm：batch方向做归一化
# LayerNorm：channel方向做归一化
# InstanceNorm： 一个channel内做归一化
# GroupNorm：将channel方向分group，
# SwitchableNorm是将BN、LN、IN结合
