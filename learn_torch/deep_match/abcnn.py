import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'abcnn'
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
        self.dropout = 0.5
        self.embedding = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim)

    def forward(self):
        pass


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
    print(model(inputs1, inputs2))