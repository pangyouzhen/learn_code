import math
import torch
import torch.nn as nn
from torch.nn.modules.transformer import Transformer


# todo
class TransformerModel(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, nhead, dim_feedforward, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = nn.Linear(embedding_dim, num_embeddings)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src: seq_length,batch_size
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        # src: seq_length,batch_size,embedding_dim
        src = self.pos_encoder(src)
        # src: seq_length,batch_size,embedding_dim
        output = self.transformer_encoder(src, self.src_mask)
        # output: seq_length,batch_size,embedding_dim
        output = self.decoder(output)
        # output: seq_length,batch_size,num_embeddings
        return output


class PositionalEncoding(nn.Module):
    # PE 层的前一层是embedding，维度是一般为 batch_size ，seq_len, embedding_dim
    # 一个句子 10个单词，每个单词 100维度表示，所以为 10 * 100
    # 参数 d_model 为 embedding的维度
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        # pe: max_len  * d_model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #  position: max_len * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # div_term: d_model / 2
        pe[:, 0::2] = torch.sin(position * div_term)
        #  max_len * (d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term)
        #  max_len * (d_model / 2)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe: max_len * 1 * d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
