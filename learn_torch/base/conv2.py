import torch.nn as nn
import torch.nn.functional as F
import torch


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'text_cnn'
        self.kernel_size = (2, 3, 4)  # 卷积核尺寸
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim,
                               kernel_size=(2, config.embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim,
                               kernel_size=(3, config.embedding_dim))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim,
                               kernel_size=(4, config.embedding_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(in_features=config.embedding_dim * len(config.kernel_size),
                            out_features=config.out_features)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(-1)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        return x

    def forward(self, input):
        #  (batch_size,seq_length)
        input_embedding = self.embedding(input)
        input_embedding = input_embedding.unsqueeze(1)
        #  (batch_size,1,seq_length,embedding_dim)
        conv1 = self.conv_and_pool(input_embedding, self.conv1)
        conv2 = self.conv_and_pool(input_embedding, self.conv2)
        conv3 = self.conv_and_pool(input_embedding, self.conv3)
        out = torch.cat((conv1, conv2, conv3), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    num_embeddings = 10
    batch_size = 1
    seq_length = 7
    embedding_dim = 5
    out_features = 2
    config = Config(num_embeddings, embedding_dim, out_features)
    model = Model(config)
    print(model)
    print("---------------------")
    for param in model.named_parameters():
        print(param[0], param[1].shape)
    inputs1 = torch.randint(high=10, size=(batch_size, seq_length))
    print(model(inputs1))
    assert model(inputs1).shape == (batch_size, out_features)
