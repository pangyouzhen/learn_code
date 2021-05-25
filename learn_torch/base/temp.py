import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 1
seq_len = 7
embedding_dim = 5
kernel = 2
out_size = 5
a = torch.rand(size=(batch_size, seq_len, embedding_dim))
b = a.unsqueeze(1)
print(b.shape)
conv1 = nn.Conv2d(in_channels=1, out_channels=1, stride=(2,), kernel_size=(kernel, embedding_dim), )
print(conv1(b).shape)
a = a.permute(0, 2, 1)
print(a.shape)
conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=4, kernel_size=(kernel,), dilation=(2,))
print(conv2(a).shape)
