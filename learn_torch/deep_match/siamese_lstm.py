import torch
import torch.nn as nn

torch.manual_seed(0)


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'siamese_lstm'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.2
        self.lr = 0.001
        self.kernel = 3


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim, )
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers, bidirectional=True,
                            batch_first=True)
        self.linear = nn.Linear(config.hidden_size * 2, config.out_features)

    def forward(self, input1, input2):
        # (batch_size, seq_length)
        input1_embedding = self.embedding(input1)
        input2_embedding = self.embedding(input2)
        # (batch_size, seq_length, embedding_dim)
        input1_lstm, (h1_hidden, h1_cell) = self.lstm(input1_embedding)
        input2_lstm, (h2_hidden, h2_cell) = self.lstm(input2_embedding)
        # (batch_size, seq_length, hidden_size *2)
        input1_lstm_output = input1_lstm[:, -1:, :]
        # (batch,1,hidden_size * 2)
        input2_lstm_output = input2_lstm[:, -1:, :]
        # (batch,1,hidden_size * 2)
        lstm_minus = torch.abs(input1_lstm_output - input2_lstm_output)
        lstm_minus = lstm_minus.squeeze(1)
        # batch_size,hidden_size * 2
        x = self.linear(lstm_minus)
        # batch_size,out_features
        return x


#

if __name__ == '__main__':
    num_embeddings = 20
    embedding_dim = 10
    hidden_size = 2
    num_layers = 2
    out_features = 2
    config = Config(num_embeddings, embedding_dim, out_features)
    lstmSiamese = Model(config)
    max_num_value = 10
    batch_size = 5
    seq_input1_length = 3
    seq_input2_length = 4
    input1 = torch.randint(max_num_value, (batch_size, seq_input1_length))
    input2 = torch.randint(max_num_value, (batch_size, seq_input2_length))
    res = lstmSiamese(input1, input2)
    print(res)
    print(res.shape)
    assert res.shape == (batch_size, out_features)
