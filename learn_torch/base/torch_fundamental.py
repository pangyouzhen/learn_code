from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learn_torch.other.PositionalEncoding import PositionalEncoding

np.random.seed(0)
torch.manual_seed(0)

#  sigmoid 函数： \frac{1}{1+e^{-x}}
#  softmax 函数： 归一化指数函数，用在多分类

# Linear: 线性层-矩阵相乘
m = nn.Linear(in_features=20, out_features=30)
n = torch.randn(50, 20)
assert m(n).size() == (50, 30)

# embedding
batch_size = 5
seq_length = 4
embedding_dim = 20
hidden_size = 30
num_class = 5
num_embeddings = 10
# 两分类

# target = np.random.randint(num_class, size=batch_size)
# target = torch.from_numpy(target).long()
target = torch.randint(num_class, size=(batch_size,))
print(target)
# embedding
x = np.random.randint(10, size=(batch_size, seq_length))
x = torch.from_numpy(x).long()
#  embedding, embedding就是lookup，寻找
# input: (*) -> output: (*,E)
print("最原始的输入:", x.size())
m = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
x_embedding = m(x)
print(x_embedding.size())
assert x_embedding.size() == (batch_size, seq_length, embedding_dim)
####################################################################################
# lstm
# https://zhuanlan.zhihu.com/p/79064602
# input: (batch_size,seq_len, embeding_dim)
lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.2)
x_lstm, _ = lstm(x_embedding)
#  (batch_size,seq_len, hidden_size)
assert x_lstm.size() == (batch_size, seq_length, hidden_size)
assert x_lstm[:, -1, :].size() == (batch_size, hidden_size)
x_lstm = x_lstm[:, -1, :]
ln = nn.Linear(in_features=hidden_size, out_features=num_class)
x_ln = ln(x_lstm)
x_softmax = torch.softmax(x_ln, dim=-1)
x_argmax = torch.argmax(x_softmax, dim=-1)
# transformer
# SNE -> TNE
# pytorch中文档：
# S：src_seq_length
# T: target_seq_length
# N: batch_size
# E: embedding_dim
####################################################################################
# PositionEncoding
# (batch_size, seq_length, embedding_dim)
posEncoding = PositionalEncoding(embedding_dim)
x_posend = posEncoding(x_embedding)
assert x_posend.size() == (batch_size, seq_length, embedding_dim)
# (batch_size, seq_length, embedding_dim)

# transformer
from torch.nn.modules.transformer import Transformer

# 内置的Transformer 是没有 embedding 和 PositionEncoding 的
encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = x_posend.permute(1, 0, 2)
out = transformer_encoder(src)
assert (out.size() == (seq_length, batch_size, embedding_dim))
out = out.permute(1, 0, 2)
# batch_size,seq_length,embedding_dim
trans_ln = nn.Linear(in_features=embedding_dim, out_features=num_embeddings)

####################################################################################
tgt_seq_length = 2
trans = Transformer(d_model=embedding_dim, nhead=4)
tgt = torch.randn(tgt_seq_length, batch_size, embedding_dim)
x_trans = trans(src, tgt)
print(x_trans.size())
assert x_trans.size() == (tgt_seq_length, batch_size, embedding_dim)

# weight = torch.Tensor([[1, 2.3, 3], [4, 5.1, 6.3]])
# embedding = nn.Embedding.from_pretrained(weight)

weight = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
                       [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]])
embedding = nn.Embedding.from_pretrained(weight)
n = torch.LongTensor([[1, 3, 4, 5], [2, 3, 6, 7]])
print(embedding(n))

# squeeze
x_embedding = torch.randn(3, 1, 4, 1)
print(x_embedding.size())
print(x_embedding.squeeze().size())

# unsqueeze
x_embedding = torch.randn(1, 4, 5)
print(x_embedding.size())
print(x_embedding.unsqueeze(2).size())
print(x_embedding.unsqueeze(1).size())
#  squeeze 挤压，就是删减维度
#  unsqueeze - squeeze的反义词，增加维度

# view
x_embedding = torch.randn(4, 4)
print(x_embedding.view(-1, 8).size())
print(x_embedding.view(-1, 8).is_contiguous())
print(x_embedding.view(-1, 8).contiguous())

# conv2d
m = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=(3,), stride=(2,))
input1 = torch.randn(20, 16, 50, 100)
print(m(input1).size())

# transpose
x_embedding = torch.randn(3, 4)
print(x_embedding.size())
print(x_embedding.transpose(1, 0).size())

#  permute
x_embedding = torch.randn(3, 4, 5, 6)
print(x_embedding.size())
print(x_embedding.permute(3, 2, 1, 0).size())

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
# 一种是mul,一种是matmul

# torch.mul() 矩阵点乘 == a*b，也就是element wise 对应位相乘
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

tensor1 = torch.randn(10, 3, 2)
tensor2 = torch.randn(4, 5, 7)
assert (torch.kron(tensor1, tensor2)).shape == (40, 15, 14)

# bmm 适用于3维，matmul 普遍适用，mm只适用二维
m = torch.randn(10, 3, 4)
n = torch.randn(10, 4, 6)
assert torch.bmm(m, n).size() == (10, 3, 6)

# dropout 主要解决过拟合的问题
m = nn.Dropout(p=0.2)
inputs = torch.randn(20, 16)
print(m(inputs).size())

# layerNorm ，对原始的输入维度没有影响
inputs = torch.randn(20, 5, 10, 10)
m = nn.LayerNorm(10)
print(m(inputs).size())

# mask fill
a = torch.tensor([4, 5, 6, 7])
masked = torch.tensor([1, 1, 0, 0]).bool()
b = a.masked_fill(mask=masked, value=torch.tensor(0))
print(b)

#  pytorch 模型的训练步骤
# 1. 数据 2. 模型 3. 损失函数 4. 优化和拟合


#  判断数据是float 还是long
import numpy as np

x_embedding = np.random.randint(10, size=(2, 3))
x_embedding = torch.from_numpy(x_embedding).long()
print(x_embedding.type())

# softmax
x_embedding = torch.rand(2, 5).int().float()
print(F.softmax(x_embedding))
print(F.softmax(x_embedding, dim=0))
print(F.softmax(x_embedding, dim=1).sum())

print(F.softmax(x_embedding, dim=1))
print(F.softmax(x_embedding, dim=1).sum())

print("激活函数-----")
print(F.tanh(x_embedding).size())

print(torch.sigmoid(x_embedding).size())

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
m = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=(3,), stride=(2,))
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
#  todo  torch 交叉熵中log softmax 那么模型的最后一层是不是不需要再进行softmax
crossentropyloss = nn.CrossEntropyLoss()
crossentropyloss_output = crossentropyloss(x_input, y_target)
print('crossentropyloss_output:\n', crossentropyloss_output)

# 常见错误
# target out of bounds, NllLoss = -x[class] 的期望，x[class] 超出索引


# m1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
# m2 = torch.nn.MaxPool1d(kernel_size=3, stride=2)
# inputm = torch.randn(2, 4, 5)
# print(m1(inputm).shape)
# print(m2(inputm).shape)
#
# m1 = torch.nn.AvgPool2d(kernel_size=3, stride=2)
# m11 = torch.nn.AvgPool2d(kernel_size=3)
# m2 = torch.nn.AvgPool1d(kernel_size=3, stride=2)
# inputm = torch.randn(2, 4, 5)
# print(m1(inputm).shape)
# print(m2(inputm).shape)
# print(m11(inputm).shape)

a = torch.randn(2, 4, 5)
print(F.max_pool1d(a))

target = torch.randn(12, 5)
out = torch.randn(12, 5)
loss = -target * torch.log(out) - (1 - target) * torch.log(1 - out)
print(loss)
loss2 = loss.sum(-1).mean()
print(loss2)
# crition = torch.nn.BCELoss()
# print(crition(target, out) == loss)


m = nn.Conv1d(in_channels=16, out_channels=33, kernel_size=(3,))
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
x_embedding = torch.randn(3, 4)
print(x_embedding)
indicies = torch.LongTensor([0, 2])
# 进行切片，根据dim和indicies 获取相关数据
print(torch.index_select(x_embedding, 0, indicies))
print(torch.index_select(x_embedding, 1, indicies))

# pytorch 归一化层：BatchNorm、LayerNorm、InstanceNorm、GroupNorm
# Norm 最归一化，所以输入输出的维度是不变化的
# BatchNorm：batch方向做归一化
# LayerNorm：channel方向做归一化
# InstanceNorm： 一个channel内做归一化
# GroupNorm：将channel方向分group，
# SwitchableNorm是将BN、LN、IN结合

## torch 随机数
# torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称

# MAE 默认是取均值
loss = nn.L1Loss()
input = torch.randn((5, 1, 3), requires_grad=True)
target = torch.randn((5, 1, 3))
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

# torch
a = torch.randint(10, size=(3, 2, 6, 7)).float()
# size 是4个元素的元组，所以一共四层[
# dim=-1 就是针对最里层的标量进行,求均值等操作就-1维度的数据削去了
# dim= 1 就是从按外层算第0，1个[
assert torch.mean(a, dim=1).size() == (3, 6, 7)
assert torch.mean(a, dim=-1).size() == (3, 2, 6)
# a = torch.randint(10, size=(3, 6, 7)).float()
# layerNorm 计算过程
#  layerNorm 针对最里层做的归一化变换
print("--------------layernorm--------------")
a = torch.FloatTensor([[[1, 2, 4, 1],
                        [6, 3, 2, 4],
                        [2, 4, 6, 1]]])
print(a.size() == (1, 3, 4))
# batch_size,seq_length,embedding
assert a[0].size() == (3, 4)
a_mean = torch.mean(a, dim=-1)
assert a_mean.size() == (1, 3)
a_var = torch.var(a, dim=-1, unbiased=False)
assert a_var.size() == (1, 3)
a_mean_expand = a_mean.unsqueeze(dim=-1).expand(1, 3, 4)
a_var_expand = torch.sqrt(a_var.unsqueeze(dim=-1).expand(1, 3, 4) + 1e-5)
res = (a - a_mean_expand) / (a_var_expand)
lm = nn.LayerNorm(4)
print(lm(a).data)
print(res.data)
print("--------------batchnorm--------------")
# TODO error
a_mean_bm = torch.mean(a, dim=0)
a_var_bm = torch.var(a, dim=0, unbiased=False)
a_mean_expand_bm = a_mean_bm.unsqueeze(dim=0).expand(1, 3, 4)
a_var_expand_bm = torch.sqrt(a_var_bm.unsqueeze(dim=0).expand(1, 3, 4) + 1e-5)
# TODO 为什么这里的维度是3, 输入默认第一个参数是batch_size
bm = nn.BatchNorm1d(3)
res2 = (a - a_mean_bm) / (a_var_expand_bm)
print(a_mean_expand_bm)
print(a_var_expand_bm)
print(bm(a).data)
print(res2.data)

print("------------------self_attention----------------")
from torch.nn.modules.transformer import MultiheadAttention

# a = torch.randint(100, size=(5, 4, 6)).float()
a = torch.randn(size=(5, 4, 6))
self_attention = MultiheadAttention(embed_dim=6, num_heads=1)
attn_output, _ = self_attention(a, a, a)
print(attn_output)
print("--------------torch index -------------")
import torch

a = torch.randint(10, size=(4, 3, 28, 28)).float()
# Pytorch风格的索引
assert a[0].size() == (3, 28, 28)
assert a[0, 0].size() == (28, 28)
assert a[0, 0, 0].size() == (28,)
print(a[0, 0, 0, 0])
print(a[0, 0, 0, 0].size())
# python风格的索引
assert torch.mean(a, dim=-1).size() == (4, 3, 28)
assert a[:2].size() == (2, 3, 28, 28)
assert a[:2, :1, :, :].shape == (2, 1, 28, 28)
assert a[:2, 1:, :, :].shape == (2, 2, 28, 28)
# 注意:1 和1是不同的
# assert a[:2, 1, :, :] == (2, 28, 28)
assert a[:2, -2:, :, :].shape == (2, 2, 28, 28)
assert a[:, :, 0:28:2, 0:28:2].shape == (4, 3, 14, 14)
assert a[:, :, ::2, ::2].shape == (4, 3, 14, 14)
# index_select（）
# 选择特定下标有时候很有用，比如上面的a这个Tensor可以看作4张RGB（3
# 通道）的MNIST图像，长宽都是28px。那么在第一维度上可以选择特定的图片，在第二维度上选择特定的通道，在第三维度上选择特定的行等：

# 选择第一张和第三张图
assert a.index_select(0, torch.tensor([0, 2])).shape == (2, 3, 28, 28)
# 选择R通道和B通道
assert a.index_select(1, torch.tensor([0, 2])).shape == (4, 2, 28, 28)
# 选择图像的0~8行
assert a.index_select(2, torch.arange(8)).shape == (4, 3, 8, 28)

# 使用 ... 索引任意多的维度
assert a[...].shape == (4, 3, 28, 28)
assert a[0, ...].shape == (3, 28, 28)
assert a[:, 1, ...].shape == (4, 28, 28)
assert a[..., :1].shape == (4, 3, 28, 1)

#  使用mask索引
import torch

a = torch.randn(3, 4)
print(a)

# 生成a这个Tensor中大于0.5的元素的掩码
mask = a.ge(0.5)
print(mask)

# 取出a这个Tensor中大于0.5的元素
# 注意：最后取出的 大于0.5的Tensor的shape已经被打平。
val = torch.masked_select(a, mask)
print(val)
print(val.shape)
# take索引是在原来Tensor的shape基础上打平，然后在打平后的Tensor上进行索引。
import torch

a = torch.tensor([[3, 7, 2], [2, 8, 3]])
print(a)
print(torch.take(a, torch.tensor([0, 1, 5])))


def attn(a, b):
    #  batch_size, seq_length, embedding
    # 交互矩阵计算
    attn = torch.matmul(a, b.transpose(2, 1))
    #  batch_size,seq_length_a,seq_length_b
    attn_a = F.softmax(attn, dim=1)
    #  batch_size,seq_length_a,seq_length_b
    attn_b = F.softmax(attn, dim=2)
    #  batch_size,seq_length_a,seq_length_b
    feature_b = torch.matmul(attn_a.transpose(1, 2), a)
    # batch_size, seq_length_b, embedding
    feature_a = torch.matmul(attn_b, b)
    return feature_a, feature_b


def conv_and_pool(x, conv):
    # x: batch_size,1,seq_length,embedding_dim
    # conv: 1, output_channel,(kernel,embedding_dim)

    # conv(x): batch_size, output_channel, seq_length - kernel_size + 1, 1
    # F.relu(conv(x)): batch_size, output_channel, seq_length - kernel_size + 1, 1
    # F.relu(conv(x)).squeeze(-1): batch_size, output_channel, seq_length - kernel_size + 1
    # F.max_pool1d(x, x.size(-1)): batch_size, output_channel, 1

    # result: batch_size, output_channel
    x = F.relu(conv(x)).squeeze(-1)
    x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
    return x


m = nn.AdaptiveMaxPool2d((5, 7))
input = torch.randn(1, 64, 8, 9)
# m = nn.ZeroPad2d((left, right, top , bottom))
m = nn.ZeroPad2d((1, 1, 2, 2))
input = torch.randn(1, 1, 3, 3)
print(m(input).shape)

import torch

batch_size = 9
a = torch.randn(size=(batch_size, 1, 6, 5))
cnn2 = nn.Conv2d(in_channels=1, out_channels=1,
                 kernel_size=(2, 5))
cnn3 = nn.Conv2d(in_channels=1, out_channels=1,
                 kernel_size=(3, 5))
cnn4 = nn.Conv2d(in_channels=1, out_channels=1,
                 kernel_size=(3, 5))
# print(cnn(a).shape)

