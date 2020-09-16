import torch
import torch.nn as nn
import torch.nn.functional as F


class Esim(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=500, hidden_size=20, num_layers=2, bidirectional=True)

    def forward(self, premise, hypothesis):
        premise_lstm = self.lstm(premise)
        hypothesis_lstm = self.lstm(hypothesis)


if __name__ == '__main__':
    a = torch.randn(2, 500)
    b = torch.randn(2, 500)
