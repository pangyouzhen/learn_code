# https://zhuanlan.zhihu.com/p/144132609
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, ):
        super(CustomLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        # i_t
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        # assumes x.shape represents (batch_size, sequence_size, input_size)
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_sz).to(x.device),
                torch.zeros(bs, self.hidden_sz).to(x.device),
            )
        else:
            h_t, c_t = init_states
        for t in range(seq_sz):
            x_t = x[:, t, :]
            #  x_t: (batch_size, input_size),  self.U_i: (input_sz, hidden_sz)
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_i + self.b_i)
            g_t = torch.sigmoid(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


if __name__ == '__main__':
    t = torch.randn(5, 2, 2)
    clstm = CustomLSTM(2, 2)
    print(clstm(t))
