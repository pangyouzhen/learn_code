import torch
import torch.nn as nn
from torch import Tensor

torch.manual_seed(1)


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.left_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True,
                                 dropout=0.2, bidirectional=True)
        self.right_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True,
                                  dropout=0.2, bidirectional=True)
        # todo 这里是否增加 requires_grad
        self.para: nn.Parameter = nn.Parameter(Tensor(1))
        self.para.data.clamp_(min=0, max=1)

    def forward(self, input):
        # batch_size, seq_length
        embed = self.embedding(input)
        # batch_size, seq_length, embedding_dim
        left_lstm, _ = self.left_lstm(embed)
        #  batch_size, seq_length, 2 * hidden_size
        right_lstm, _ = self.right_lstm(embed)
        #  batch_size, seq_length, 2 * hidden_size
        temp_left = left_lstm[:, -1]
        temp_right = right_lstm[:, -1]
        end = self.para * temp_left + (1 - self.para) * temp_right
        return end


if __name__ == '__main__':
    model = Model(num_embeddings=200, embedding_dim=5, hidden_size=4)
    input = torch.randint(low=0, high=10, size=(3, 2))
    res = model(input)
    print(res)
