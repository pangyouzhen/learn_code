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
        input1_embedding = self.embedding(input1)
        input2_embedding = self.embedding(input2)
        input1_lstm, (h1_hidden, h1_cell) = self.lstm(input1_embedding)
        input2_lstm, (h2_hidden, h2_cell) = self.lstm(input2_embedding)
        dist = torch.cdist(h1_hidden, h2_hidden, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        return dist


if __name__ == '__main__':
    lstmSiamese = LstmSiamese(num_embeddings=20, embedding_dim=10, hidden_size=2, num_layers=2)
    input1 = torch.randn(10, 4).long()
    print(input1)
    # print(input1.type())
    input2 = torch.randn(10, 3).long()
    # print(input2.type())
    print(input2)
    print(lstmSiamese(input1, input2))
