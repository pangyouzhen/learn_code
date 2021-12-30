import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'siamese_cnn'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.2
        self.lr = 0.001
        self.kernel = 3
        self.kernel_size = (2, 3, 4)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim, )
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim, kernel_size=(2, config.embedding_dim))
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim, kernel_size=(3, config.embedding_dim))
        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=config.embedding_dim, kernel_size=(4, config.embedding_dim))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features=config.embedding_dim * len(config.kernel_size),
                            out_features=config.out_features)

    def conv_and_pool(self, x, conv):
        # x: batch_size,1,seq_len,embedding
        # conv: 1, embedding, (kernel,embedding)
        x = F.relu(conv(x)).squeeze(-1)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        # batch_size,embedding
        return x

    def forward(self, x1, x2):
        # (batch_size, seq_length)
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        # (batch_size,1, seq_length, embedding_dim)
        x1_pool_1 = self.conv_and_pool(x1, self.cnn1)
        # batch_size,embedding
        x1_pool_2 = self.conv_and_pool(x1, self.cnn2)
        # batch_size,embedding
        x1_pool_3 = self.conv_and_pool(x1, self.cnn3)
        # batch_size,embedding

        # (batch_size,1, seq_length, embedding_dim)
        x2_pool_1 = self.conv_and_pool(x2, self.cnn1)
        # batch_size,embedding
        x2_pool_2 = self.conv_and_pool(x2, self.cnn2)
        # batch_size,embedding
        x2_pool_3 = self.conv_and_pool(x2, self.cnn3)
        # batch_size,embedding

        out1 = torch.cat((x1_pool_1, x1_pool_2, x1_pool_3), dim=1)
        # batch_size, 3 * embedding

        out2 = torch.cat((x2_pool_1, x2_pool_2, x2_pool_3), dim=1)
        # batch_size, 3 * embedding
        out = torch.abs(out1 - out2)
        # # batch_size, 3 * embedding
        out = self.fc(out)
        return out


#

if __name__ == '__main__':
    num_embeddings = 100
    embedding_dim = 10
    hidden_size = 2
    num_layers = 2
    out_features = 2
    config = Config(num_embeddings, embedding_dim, out_features)
    lstm_cnn = Model(config)
    max_num_value = 100
    batch_size = 5
    seq_input1_length = 10
    seq_input2_length = 15
    input1 = torch.randint(max_num_value, (batch_size, seq_input1_length))
    input2 = torch.randint(max_num_value, (batch_size, seq_input2_length))
    res = lstm_cnn(input1, input2)
    assert res.shape == (batch_size, out_features)
