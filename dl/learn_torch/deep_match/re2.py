import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, ModuleDict, Conv1d, Linear
from typing import Collection


# todo
class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'esim'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001
        self.blocks = 3
        self.enc_layers = 2
        self.kernel_sizes = (3,)
        self.num_classes = 2


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.embedding = Embedding(config)
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(config, config.embedding_dim if i == 0 else config.embedding_dim + config.hidden_size),
            'alignment': MappedAlignment(
                config,
                config.embedding_dim + config.hidden_size if i == 0 else config.embedding_dim + config.hidden_size * 2),
            'fusion': FullFusion(
                config,
                config.embedding_dim + config.hidden_size if i == 0 else config.embedding_dim + config.hidden_size * 2),
        }) for i in range(config.blocks)])
        self.connection = AugmentedResidual()
        self.pooling = Pooling()
        self.prediction = AdvancedPrediction(config)

    def forward(self, inputs):
        a = inputs['text1']
        b = inputs['text2']
        mask_a = inputs['mask1']
        mask_b = inputs['mask2']

        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        return self.prediction(a, b)


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim, padding_idx=0)
        self.dropout = config.dropout

    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, self.dropout, self.training)
        return x


class Alignment(nn.Module):
    def __init__(self, config, __):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(config.hidden_size)))

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        attn.masked_fill_(~mask, -1e7)
        attn_a = F.softmax(attn, dim=1)
        attn_b = F.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        self.add_summary('temperature', self.temperature)
        self.add_summary('attention_a', attn_a)
        self.add_summary('attention_b', attn_b)
        return feature_a, feature_b


class Encoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.dropout = config.dropout
        self.encoders = nn.ModuleList([NewConv1d(
            in_channels=input_size if i == 0 else config.hidden_size,
            out_channels=config.hidden_size,
            kernel_sizes=config.kernel_sizes) for i in range(config.enc_layers)])

    def forward(self, x, mask):
        x = x.transpose(1, 2)  # B x C x L
        mask = mask.transpose(1, 2)
        for i, encoder in enumerate(self.encoders):
            x.masked_fill_(~mask, 0.)
            if i > 0:
                x = F.dropout(x, self.dropout, self.training)
            x = encoder(x)
        x = F.dropout(x, self.dropout, self.training)
        return x.transpose(1, 2)  # B x L x C


class MappedAlignment(Alignment):
    def __init__(self, config, input_size):
        super().__init__(config, input_size)
        self.projection = nn.Sequential(
            nn.Dropout(config.dropout),
            NewLinear(input_size, config.hidden_size, activations=True),
        )

    def _attention(self, a, b):
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)


class Pooling(nn.Module):
    def forward(self, x, mask):
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class NewLinear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class FullFusion(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.dropout = config.dropout
        self.fusion1 = NewLinear(input_size * 2, config.hidden_size, activations=True)
        self.fusion2 = NewLinear(input_size * 2, config.hidden_size, activations=True)
        self.fusion3 = NewLinear(input_size * 2, config.hidden_size, activations=True)
        self.fusion = NewLinear(config.hidden_size * 3, config.hidden_size, activations=True)

    def forward(self, x, align):
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = F.dropout(x, self.dropout, self.training)
        return self.fusion(x)


class AugmentedResidual(nn.Module):
    def forward(self, x, res, i):
        if i == 1:
            return torch.cat([x, res], dim=-1)  # res is embedding
        hidden_size = x.size(-1)
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)  # latter half of res is embedding


class Prediction(nn.Module):
    def __init__(self, config, inp_features=2):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(config.dropout),
            NewLinear(config.hidden_size * inp_features, config.hidden_size, activations=True),
            nn.Dropout(config.dropout),
            Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))


class AdvancedPrediction(Prediction):
    def __init__(self, config):
        super().__init__(config, inp_features=4)

    def forward(self, a, b):
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


class NewConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes: Collection[int]):
        super().__init__()
        assert all(k % 2 == 1 for k in kernel_sizes), 'only support odd kernel sizes'
        assert out_channels % len(kernel_sizes) == 0, 'out channels must be dividable by kernels'
        out_channels = out_channels // len(kernel_sizes)
        convs = []
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=(kernel_size - 1) // 2)
            nn.init.normal_(conv.weight, std=math.sqrt(2. / (in_channels * kernel_size)))
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(nn.utils.weight_norm(conv), GeLU()))
        self.model = nn.ModuleList(convs)

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.model], dim=-1)


def process_data(batch, device=torch.device("cpu")):
    text1 = torch.LongTensor(batch['text1']).to(device)
    text2 = torch.LongTensor(batch['text2']).to(device)
    mask1 = torch.ne(text1, 0).unsqueeze(2)
    mask2 = torch.ne(text2, 0).unsqueeze(2)
    inputs = {
        'text1': text1,
        'text2': text2,
        'mask1': mask1,
        'mask2': mask2,
    }
    if 'target' in batch:
        target = torch.LongTensor(batch['target']).to(device)
        return inputs, target
    return inputs, None


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
    inputs1 = torch.randint(high=50, size=(batch_size, max_length))
    inputs2 = torch.randint(high=50, size=(batch_size, max_length))
    target = torch.randint(high=2, size=(batch_size,))
    batch = {"text1": inputs1, "text2": inputs2, "target": target}
    inputs, target = process_data(batch)
    res = model(inputs)
    assert res.shape == (batch_size, out_features)
