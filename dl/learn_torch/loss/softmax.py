import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_func(x:torch.Tensor,dim:int=1):
    exp_val = torch.exp(x)
    return exp_val / torch.sum(exp_val,dim,keepdim=True)

class SoftmaxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x:torch.Tensor,dim:int=1):
        x = torch.exp(x)
        xs = torch.sum(x,dim=dim,keepdim=True)
        div_val = x / xs
        return torch.log(div_val).sum(dim) / x.shape[dim]
    

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(3,10)
    print(f"{x = }")
    s1 = nn.Softmax(dim=1)
    s1_result = s1(x)
    print(f"{s1_result.shape}")
    s2 = softmax_func(x)
    print(s1_result)
    # 这里每个值都相等，但是为啥equal不相等
    # assert torch.equal(softmax_func(x),s1_result) is True
        