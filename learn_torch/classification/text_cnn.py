import torch.nn as nn
import torch.nn.functional as F
import torch


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'text_cnn'
        self.kernel_size = 2  # 卷积核尺寸
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim,
                             kernel_size=(config.kernel_size, config.embedding_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(in_features=config.embedding_dim, out_features=config.out_features)

    def forward(self, input):
        #  (batch_size,seq_length)
        input_embedding = self.embedding(input)
        input_embedding = input_embedding.unsqueeze(1)
        #  (batch_size,1,seq_length,embedding_dim)
        cnn_encoding = self.cnn(input_embedding)
        # 这里输入的最后两个维度 seq_length,embedding_dim 和 kernel_size 的维度对应计算
        # (batch_size,embedding_dim,seq_length - kernel_size + 1, embedding_dim - embedding_dim +1)
        cnn_encoding = F.relu(cnn_encoding)
        cnn_encoding = cnn_encoding.squeeze(-1)
        # (batch_size,embedding_dim, seq_length - kernel_size + 1)
        cnn = F.max_pool1d(cnn_encoding, cnn_encoding.size(-1))
        # (batch_size, embedding_dim, 1)
        cnn = cnn.squeeze(-1)
        # (batch_size, embedding_dim)
        out = self.dropout(cnn)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    num_embeddings = 5000
    batch_size = 128
    seq_length = 5
    embedding_dim = 300
    out_features = 10
    config = Config(num_embeddings, embedding_dim, out_features)
    model = Model(config)
    print(model)
    print("---------------------")
    for param in model.named_parameters():
        print(param[0], param[1].shape)
    inputs1 = torch.randint(high=200, size=(batch_size, seq_length))
    print(model(inputs1))
