from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)

# ！！！！！
#  输入：input: batch_size ,seq_length，
#  embedding:
#       定义： nn.Embedding(vocab_size, embedding_dim)
#       输出: batch_size,seq_length,embedding_dim
#  encoding:
#       LSTM
#           定义: nn.LSTM(embedding_dim, hidden_dim ,num_layers=1, bidirectional=*)
#               forward 可以随机初始化 h0，c0 (num_layers,seq_length,hidden_dim)
#           输出：lstm: seq_length，batch_size, hidden_size 或者是 2*hidden_size
#       transformer:
#           embedding 之后需要经过 PositionEncoding(d_model = embeding_dim) 维度不变（batch_size,seq_length,embedding）
#           定义 transformer(d_model=embedding_dim, nhead= embedding 的整数倍)
#           输出： tgt_excepted_size, seq_length, embedding_dim
#       cnn:
#
#  sigmoid 函数： \frac{1}{1+e^{-x}}
#  softmax 函数： 归一化指数函数，用在多分类

# Linear: 线性层-矩阵相乘
m = nn.Linear(in_features=20, out_features=30)
n = torch.randn(50, 20)
assert m(n).size() == (50, 30)

# transformer
# SNE -> TNE
# pytorch中文档：
# S：src_seq_length
# T: target_seq_length
# N: batch_size
# E: embedding_dim
from torch.nn.modules.transformer import Transformer

# 内置的Transformer 是没有 embedding 和 PositionEncoding 的
trans = Transformer(d_model=10, nhead=2)
src = torch.randn(20, 32, 10)
tgt = torch.randn(10, 32, 10)
assert trans(src, tgt).size() == (10, 32, 10)

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
assert (out.size() == (10, 32, 512))

# lstm
# https://zhuanlan.zhihu.com/p/79064602
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
input1 = torch.randn(5, 3, 10)
# input: (seq_len, batch_size, embeding_dim)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (h0, c0) = lstm(input1, (h0, c0))
output_1, (h0_1, c0_1) = lstm(input1)
print(output_1.size())
print(h0_1.size())
print(c0_1.size())
assert output.size() == (5, 3, 20)
# TODO
# 这里究竟用output 还是用 h0
# h_n of shape (num_layers * num_directions, batch, hidden_size)
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

weight = torch.Tensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
input = torch.LongTensor([1])
print(embedding(input))
# assert embedding(input) == torch.Tensor([[4, 5.1, 6.3]])

weight = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
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

#  max softmax
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
print(torch.max(a, dim=1))
print(type(torch.max(a, dim=1)))
print(torch.max(a, dim=1).indices)

## pytorch 两种矩阵相乘的方式

# torch.mul() 矩阵点乘 == a *b，也就是element wise 对应位相乘
# torch.matmul() 矩阵相乘，对应 python 中 x @ y  \otimes

# element wise --- Hadamard product.  \odot
m = torch.Tensor([[1, 2, 3], [4, 5, 6]])
n = torch.Tensor([[4, 5, 6], [1, 7, 3]])
assert bool((torch.mul(m, n) == m * n).all()) == True
print(torch.mul(m, n).size())

# tensor product 也是矩阵相乘: \otimes
# 同时也和 torch.bmm 相等
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
assert bool((torch.matmul(tensor1, tensor2) == (tensor1 @ tensor2)).all()) == True
print(torch.matmul(tensor1, tensor2).size())

# bmm 适用于3维，matmul 普遍适用，mm只适用二维
m = torch.randn(10, 3, 4)
n = torch.randn(10, 4, 6)
assert torch.bmm(m, n).size() == (10, 3, 6)

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

# https://www.zhihu.com/question/294679135
x_input = torch.randn(3, 3)  # 随机生成输入
y_target = torch.tensor([1, 2, 0])  # 设置输出具体值 print('y_target\n',y_target)

# 直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
crossentropyloss = nn.CrossEntropyLoss()
crossentropyloss_output = crossentropyloss(x_input, y_target)
print('crossentropyloss_output:\n', crossentropyloss_output)

# 常见错误
# target out of bounds, NllLoss = -x[class] 的期望，x[class] 超出索引


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

# 切片
x = torch.randn(3, 4)
print(x)
indicies = torch.LongTensor([0, 2])
# 进行切片，根据dim和indicies 获取相关数据
print(torch.index_select(x, 0, indicies))
print(torch.index_select(x, 1, indicies))

# pytorch 归一化层：BatchNorm、LayerNorm、InstanceNorm、GroupNorm
# Norm 最归一化，所以输入输出的维度是不变化的
# BatchNorm：batch方向做归一化
# LayerNorm：channel方向做归一化
# InstanceNorm： 一个channel内做归一化
# GroupNorm：将channel方向分group，
# SwitchableNorm是将BN、LN、IN结合

## torch 随机数
#  生成标准正太分布，所以转换long完成后 会有负数，没法直接输入到embedding中
print(torch.randn(10, 2))
# 生成[0,1)之间的均匀分布，所以转换long完成后都是0值
print(torch.rand(10, 3))
#  生成 0-10 之间的整数
print(torch.randint(10, (2, 2)).type())
# 因为embedding层 是根据索引去找，所以是需要传入的是longtensor
# Longtensor 是64位的整数
# Inttensor 是32位的整数
# torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称

# MAE 默认是取均值
loss = nn.L1Loss()
input = torch.randn((5, 1, 3), requires_grad=True)
target = torch.randn((4, 1, 3))
output = loss(input, target)
output.backward()

# 常见错误 RuntimeError: Boolean value of Tensor with more than one value is ambiguous
# 这个问题的原因是 最常見的地方在於 Loss Function 的使用，通常是誤用了 Class 作為輸入。如果得到了這個報錯，可以檢查是否 Loss function 的地方有誤用。

from torch.nn import init

linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
