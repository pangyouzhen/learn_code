# https://www.cnblogs.com/xiximayou/p/13343856.html
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from learn_torch.transformer_ import clones


# 直接对应公式即可
def attention(query, key, value):
    # dk为词向量的维度。其目的在于调节的作用，使得内积不易过大。
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1, ):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head == 0
        self.d_ = d_model // head
        self.h = head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        print(query.size())
        # TODO 这里的compl 怎么理解，应该是输出一个tensor？
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        print(query.size())
        x, self.attn = attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_)
        return self.linears[-1](x)


if __name__ == '__main__':
    # 输入 batch_size,seq_length,embedding_dim
    query = torch.randint(10, [2, 3, 8]).float()
    key = torch.randint(10, [2, 4, 8]).float()
    value = torch.randint(10, [2, 4, 8]).float()
    # res, p_attn = attention(query, key, value)
    # print(res.shape, p_attn.shape)
    mha = MultiHeadAttention(4, 8)
    mha(query, key, value)
