import torch
import torch.nn as nn
import torch.nn.functional as F


# todo siamese 效果这么差
class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'siamese_cbow'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001
        self.kernel = 3


class CBOW(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.linear1 = nn.Linear(32 * embedding_dim, 128)
        self.linear2 = nn.Linear(128, num_embeddings)

    def forward(self, x):
        # batch_size,seq_length
        x = self.embedding(x)
        # batch_size,seq_length,embedding_dim
        x = x.view(x.shape[0], 1, -1)
        # batch_size,1, seq_length * embedding_dim,
        x = x.squeeze(1)
        # batch_size, seq_length * embedding_dim,
        x = self.linear1(x)
        # batch_size, 128
        x = F.relu(x, inplace=True)
        # batch_size, 128
        x = self.linear2(x)
        # batch_zie, num_embeddings
        return x


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.dropout = 0.5
        self.cbow = CBOW(config.num_embeddings, config.embedding_dim)
        self.linear = nn.Linear(config.num_embeddings, config.out_features)

    def forward(self, s1, s2):
        # batch_size,seq_length
        s1_cbow = self.cbow(s1)
        # batch_zie, num_embeddings
        s2_cbow = self.cbow(s2)
        x = torch.abs(s1_cbow - s2_cbow)
        # batch_zie, num_embeddings
        x = self.linear(x)
        return x


if __name__ == '__main__':
    num_embeddings = 5000
    batch_size = 128
    seq_length = 4
    embedding_dim = 300
    out_features = 10
    epoch = 5
    config = Config(num_embeddings, embedding_dim, seq_length, out_features)
    model = Model(config)
    print(model)
    print("---------------------")
    for param in model.named_parameters():
        print(param[0], param[1].shape)
    inputs1 = torch.randint(high=200, size=(batch_size, seq_length))
    inputs2 = torch.randint(high=200, size=(batch_size, seq_length))
    res = (model(inputs1, inputs2))
    print(res.shape)
    assert res.shape == (batch_size, out_features)
