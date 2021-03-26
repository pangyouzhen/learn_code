import torch
import torch.nn as nn

torch.manual_seed(0)


class LstmSiamese(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers):
        super(LstmSiamese, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, )
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True)

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
        temp = torch.abs(input1_lstm_output - input2_lstm_output)
        return torch.sum(temp, dim=-1)


if __name__ == '__main__':
    lstmSiamese = LstmSiamese(num_embeddings=20, embedding_dim=10, hidden_size=2, num_layers=2)
    max_num_value = 10
    batch_size = 5
    seq_input1_length = 3
    seq_input2_length = 4
    input1 = torch.randint(max_num_value, (batch_size, seq_input1_length))
    input2 = torch.randint(max_num_value, (batch_size, seq_input2_length))
    res = lstmSiamese(input1, input2)
    assert res.size() == (5, 1)
