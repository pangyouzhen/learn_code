import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder,TransformerEncoderLayer
from transformers import ViTModel


def img2embed_naive(image,patch_size,weight):
    # 这里相当于卷积，直接使用unfold拿出卷积块
    patch = F.unfold(image,kernel_size=patch_size,stride=patch_size)
    pass


def img2embed_conv():
    pass
class Vit(nn.Module):
    def __init__(self,linear_dim:None=512,category:int =10) -> None:
        super().__init__()
        self.linear = nn.Linear(196,linear_dim)
        self.encoderlayer = TransformerEncoderLayer(d_model=512,nhead=8)
        self.encoder = TransformerEncoder(self.encoderlayer,num_layers=8)
        self.category = nn.Linear(linear_dim,category)
        self.embedding = nn.Embedding(1000,768)
    
    def forward(self,x:torch.Tensor):
        B,C,H,W = x.shape
        assert H == 224
        assert W == 224
        x = x.flatten(2)
        # x = torch.split(224,dim=2)
        # x = torch.cat(x.split(16,dim=2),dim=2)
        # x = torch.cat(x.split(16,dim=3),dim=3)
        assert x.shape[2] == 14
        assert x.shape[3] == 14
        x = torch.view(B,C,-1)
        x = self.linear(x)
        y = torch.zeros(B,1,196)
        x = torch.cat(x,y,dim=1)
        x = x[:,0,...]
        x = self.encoder(x)
        x = self.category(x)
        return x
    
if __name__ == "__main__":
    vit = Vit()
    model_dim = 8
    x = torch.randn(10,3,224,224)
    weight = torch.randn(None,model_dim)
    img2embed_naive(x,patch_size=4,)
    # print(vit(x).shape) 