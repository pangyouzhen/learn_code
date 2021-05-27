import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, num_embeddings, embedding_dim, context_size, ):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, num_embeddings)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(1, -1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        # x = F.softmax(x)
        return x


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.dropout = 0.5
        self.cbow = CBOW(config.num_embeddings, config.num_embeddings, 2)

    def forward(self, s1, s2):
        s1_cbow = self.cbow(s1)
        s2_cbow = self.cbow(s2)
        temp = torch.abs(s1_cbow - s2_cbow)
        return torch.sum(temp, dim=-1)


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
