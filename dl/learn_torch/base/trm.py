import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.modules.activation import MultiheadAttention

def scaled_dot_product_attention(query:Tensor,key:Tensor,value:Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1,2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp/scale,dim=-1)
    return softmax.bmm(value)

class AttentionHead(nn.Module):
    def __init__(self,dim_in:int,dim_q:int,dim_k:int) -> None:
        super().__init__()
        self.q = nn.Linear(dim_in,dim_q)
        self.k = nn.Linear(dim_in,dim_k)
        self.v = nn.Linear(dim_in,dim_k)
        
    def forward(self,query:Tensor,key:Tensor,value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query),self.k(key),self.v(value))


if __name__ == '__main__':
    # trm_encoder = TrmEncoder(d_model=512, nhead=8)
    # ma = MA(512,8)
    bulitin_multiha = MultiheadAttention(512,8)
    src = torch.rand(10, 32, 512)
    print("raw_shape",src.shape)
    # ma_res = ma(src,)
    print("not built-in----",ma_res[0].shape,ma_res[1].shape)
    multiha_res = bulitin_multiha(src,src,src)
    print("builtin shape+++++",multiha_res[0].shape,multiha_res[1].shape)
