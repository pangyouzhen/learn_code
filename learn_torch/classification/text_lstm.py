import torch
import torch.nn as nn


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'text_lstm'
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
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.lstm = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.hidden_size,
                            num_layers=config.num_layer,
                            batch_first=True,
                            dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(in_features=config.hidden_size * 2, out_features=config.out_features)

    def forward(self, sentence):
        # batch_size, seq_length
        x = self.embedding(sentence)
        # batch_size, seq_length, embedding_dim
        x, _ = self.lstm(x)
        # batch_size,seq_length,hidden_size
        out = x[:, -1, :]
        # batch_size,1,hidden_size
        pre_label = torch.softmax(self.linear(out), dim=-1)
        return pre_label
