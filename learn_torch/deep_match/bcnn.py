import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'bcnn'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001
        self.kernel = 3


# todo
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.dropout = 0.5
        self.embedding = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim,
                               kernel_size=(config.kernel, config.embedding_dim), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim,
                               kernel_size=(config.kernel, config.embedding_dim), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=config.out_features, out_channels=config.out_features,
                               kernel_size=config.kernel)
        self.conv4 = nn.Conv2d(in_channels=config.out_features, out_channels=config.out_features,
                               kernel_size=config.kernel)

    def forward(self, a, b):
        # [batch_size,seq_length]
        a_embedding = self.embedding(a)
        b_embedding = self.embedding(b)
        # [batch_size, seq_length, embedding_dim]
        # 进行宽卷积
        a_embedding = a_embedding.unsqueeze(1)
        b_embedding = b_embedding.unsqueeze(1)
        a_conv1 = self.conv1(a_embedding)
        b_conv1 = self.conv2(b_embedding)
        a_conv1 = a_conv1.squeeze(-1)
        b_conv1 = b_conv1.squeeze(-1)
        # w-ap 区别于后面的all-ap
        a_w_ap = F.avg_pool1d(a_conv1, kernel_size=a_conv1.size(-1))
        b_w_ap = F.avg_pool1d(b_conv1, kernel_size=b_conv1.size(-1))
        #
        a_conv2 = self.conv2(a_w_ap)
        b_conv2 = self.conv2(b_w_ap)
        # all-ap
        a_all_ap = F.avg_pool2d(a_conv2)
        b_all_ap = F.avg_pool2d(b_conv2)
        return F.logsigmoid(a_all_ap, b_all_ap)


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
    inputs1 = torch.randint(high=200, size=(batch_size, 5))
    inputs2 = torch.randint(high=200, size=(batch_size, 7))
    print(model(inputs1, inputs2))
