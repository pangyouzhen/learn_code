import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'match_pyramid'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.kernel_size = 2
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001
        self.max_seq_len = 32
        self.conv1_size = "5_5_8"
        self.pool1_size = "10_10"
        self.conv2_size = "3_3_16"
        self.pool2_size = "5_5"
        self.mp_hidden = 128


class Model(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.max_len1 = config.max_seq_len
        self.max_len2 = config.max_seq_len
        self.conv1_size = [int(_) for _ in config.conv1_size.split("_")]
        self.pool1_size = [int(_) for _ in config.pool1_size.split("_")]
        self.conv2_size = [int(_) for _ in config.conv2_size.split("_")]
        self.pool2_size = [int(_) for _ in config.pool2_size.split("_")]
        self.dim_hidden = config.mp_hidden

        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.conv1_size[-1],
                                     kernel_size=tuple(
                                         self.conv1_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1_size[-1],
                                     out_channels=self.conv2_size[-1],
                                     kernel_size=tuple(
                                         self.conv2_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        self.pool1 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool1_size))
        self.pool2 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool2_size))
        self.linear1 = torch.nn.Linear(self.pool2_size[0] * self.pool2_size[1] * self.conv2_size[-1],
                                       self.dim_hidden, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = torch.nn.Linear(self.dim_hidden, config.out_features, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear2.weight)

    def forward(self, x1, x2):
        # x1,x2:[batch, seq_len]
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        bs, seq_len1, embedding_dim = x1.size()
        seq_len2 = x2.size()[1]
        pad1 = self.max_len1 - seq_len1
        pad2 = self.max_len2 - seq_len2
        # simi_img:[batch, 1, seq_len, seq_len]
        # x1_norm = x1.norm(dim=-1, keepdim=True)
        # x1_norm = x1_norm + 1e-8
        # x2_norm = x2.norm(dim=-1, keepdim=True)
        # x2_norm = x2_norm + 1e-8
        # x1 = x1 / x1_norm
        # x2 = x2 / x2_norm
        # use cosine similarity since dim is too big for dot-product
        # todo 这里和原始的模型不太一样 这里直接交互矩阵
        simi_img = torch.matmul(x1, x2.transpose(1, 2)) / np.sqrt(embedding_dim)
        if pad1 != 0 or pad2 != 0:
            simi_img = F.pad(simi_img, (0, pad2, 0, pad1))
        assert simi_img.size() == (bs, self.max_len1, self.max_len2)
        simi_img = simi_img.unsqueeze(1)
        # self.logger.info(simi_img.size())
        # [batch, 1, conv1_w, conv1_h]
        simi_img = F.relu(self.conv1(simi_img))
        # [batch, 1, pool1_w, pool1_h]
        simi_img = self.pool1(simi_img)
        # [batch, 1, conv2_w, conv2_h]
        simi_img = F.relu(self.conv2(simi_img))
        # # [batch, 1, pool2_w, pool2_h]
        simi_img = self.pool2(simi_img)
        # assert simi_img.size()[1] == 1
        # [batch, pool1_w * pool1_h * conv2_out]
        simi_img = simi_img.squeeze(1).view(bs, -1)
        # output = self.linear1(simi_img)
        output = self.linear2(F.relu(self.linear1(simi_img)))
        return output


if __name__ == '__main__':
    num_embeddings = 5000
    batch_size = 128
    max_length = 32
    embedding_dim = 300
    out_features = 10
    epoch = 5
    config = Config(num_embeddings, embedding_dim, out_features)
    model = Model(config)
    print(model)
    print("---------------------")
    for param in model.named_parameters():
        print(param[0], param[1].shape)
    inputs1 = torch.randint(high=200, size=(batch_size, max_length))
    inputs2 = torch.randint(high=200, size=(batch_size, max_length))
    res = (model(inputs1, inputs2))
    assert res.shape == (batch_size, out_features)
    print(f"输出的维度是{res.shape}")
