import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
# batch_size,seq_length,embedding
a = torch.randint(100, size=(3, 2, 4))
a = a.float()
# print(a)
# tensor([[[44., 39., 33., 60.],
#          [63., 79., 27.,  3.]],
#         [[97., 83.,  1., 66.],
#          [56., 99., 78., 76.]],
#         [[56., 68., 94., 33.],
#          [26., 19., 91., 54.]]])

ln = nn.LayerNorm(4)
t = ln(a)
print(torch.mean(t, dim=-1))
print(torch.var(t, dim=-1, unbiased=False))

ln = nn.LayerNorm([2, 4])
t = ln(a)
# shape= 3 * 1
print(torch.mean(t, dim=[-2, -1]))
# shape = 3 * 1
print(torch.var(t, dim=[-2, -1], unbiased=False))

bn = nn.BatchNorm1d(2)
t = bn(a)
print(torch.mean(t, dim=[0, 2]))
print(torch.var(t, dim=[0, 2], unbiased=False))

# a = torch.Tensor([3, 4])
# torch 中这里是不能直接输入一个向量的
# a = a.unsqueeze(0).float()
a = torch.randn(3, 4)
b = torch.randn(3, 4)
s = F.cosine_similarity(a, b)
assert s.shape == (3,)

a = torch.randn(3, 4, 5)
b = torch.randn(3, 4, 5)
s = F.cosine_similarity(a, b, dim=-1)
assert s.shape == (3, 4)

a = torch.randn(3, 4, 5)
b = torch.randn(3, 4, 5)
s = F.cosine_similarity(a, b)
assert s.shape == (3, 5)
