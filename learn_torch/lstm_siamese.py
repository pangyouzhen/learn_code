import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class LstmSiamese(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers):
        super(LstmSiamese, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, )
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True)

    def forward(self, input1, input2):
        # (batch_size, seq_length)
        input1_embedding = self.embedding(input1)
        input2_embedding = self.embedding(input2)
        # (batch_size, seq_length, embedding_dim)
        input1_embedding = input1_embedding.transpose(0, 1)
        input2_embedding = input2_embedding.transpose(0, 1)
        # (seq_length,batch_size)
        input1_lstm, (h1_hidden, h1_cell) = self.lstm(input1_embedding)
        input2_lstm, (h2_hidden, h2_cell) = self.lstm(input2_embedding)
        input1_lstm = input1_lstm.transpose(0, 1)
        input2_lstm = input2_lstm.transpose(0, 1)
        # (batch_size, seq_length, hidden_size *2)
        print(input1_lstm[:, -1:, :].size())
        print(input2_lstm[:, -1:, :].size())
        loss = nn.L1Loss()
        dist = loss(input1_lstm[:, -1:, :], input2_lstm[:, -1:, :])
        return dist


if __name__ == '__main__':
    lstmSiamese = LstmSiamese(num_embeddings=20, embedding_dim=10, hidden_size=2, num_layers=2)
    input1 = torch.randint(10, (5, 3))
    input2 = torch.randint(10, (5, 4))
    print(lstmSiamese(input1, input2))
