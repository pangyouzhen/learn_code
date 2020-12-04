import torch
import torchsnooper


@torchsnooper.snoop()
def myfunc(mask, x):
    y = torch.zeros(6, device="cuda")
    y.masked_scatter_(mask, x)
    return y


if __name__ == '__main__':
    mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda', dtype=torch.uint8)
    source = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = myfunc(mask, source)
