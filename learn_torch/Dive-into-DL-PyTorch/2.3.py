import torch

# 注意在y.backward()时，如果y是标量，则不需要为backward()传入任何参数；否则，需要传入一个与y同形的Tensor。解释见 2.3.2 节。

x: torch.Tensor = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y: torch.Tensor = x + 2
print(y)
print(y.grad_fn)
print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()
print(z, out)

# 通过.requires_grad_()来用in-place的方式改变requires_grad属性：
a = torch.randn(2, 2)
# 默认是false
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)  # True
b = (a * a).sum()
print(b.grad_fn)
print("----------------")
out.backward()
print(x.grad)
# TODO 这个手动算一下，尤其是\sum符号很容易搞错，解决\sum 的最好方式是展开
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])

out2 = x.sum()
print(out2)
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)
