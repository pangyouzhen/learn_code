# https://www.cnblogs.com/xiximayou/p/13343856.html
import math

import torch
import torch.nn as nn


# 直接对应公式即可
class Attention(nn.Module):
    # dk为词向量的维度。其目的在于调节的作用，使得内积不易过大。
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        return torch.matmul(torch.softmax(scores, dim=-1), value)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, embed_dim, dropout=0.1, ):
        # embed_dim 其实是d_model
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert embed_dim % num_head == 0
        self.dim_head = embed_dim // num_head
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        self.attention = Attention()
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = query.size(0)
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        context = self.attention(Q, K, V)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        # 残差
        out = out + x
        out = self.layer_norm(out)
        return out


if __name__ == '__main__':
    # 输入 batch_size,seq_length,embedding_dim
    query = torch.randint(10, [2, 3, 8]).float()
    key = torch.randint(10, [2, 4, 8]).float()
    value = torch.randint(10, [2, 4, 8]).float()
    mha = MultiHeadAttention(4, 8)
    print(mha(query))
