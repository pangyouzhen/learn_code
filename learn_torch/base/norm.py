import torch
import torch.nn as nn

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

a_mean = torch.mean(a, dim=-1)
a_var = torch.var(a, dim=-1, unbiased=False)
a_mean_expand = a_mean.unsqueeze(dim=-1).expand(3, 2, 4)
a_var_expand = torch.sqrt(a_var.unsqueeze(dim=-1).expand(3, 2, 4) + 1e-5)
res = (a - a_mean_expand) / (a_var_expand)
lm = nn.LayerNorm(4)
print(lm(a).data)
print(res.data)
print(torch.mean(res.data, dim=-1))

ln = nn.LayerNorm([2, 4])
print(ln(a))
# todo 这里不明白是怎么计算出来的
print(torch.mean(ln(a), dim=-1))
