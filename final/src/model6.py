import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class MLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_drop_targets = []

        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.mx = nn.Linear(input_size, hidden_size)
        self.mh = nn.Linear(hidden_size, hidden_size)

        self.h0 = nn.Parameter(torch.zeros(hidden_size))
        self.c0 = nn.Parameter(torch.zeros(hidden_size))

        for name, param in list(self.cell.named_parameters()):
            if 'weight' in name:
                nn.init.orthogonal(param.data)
                self.register_weight_drop(self.cell, name, 0.5)
            elif 'bias' in name:
                param.data[hidden_size:hidden_size * 2] = 1
        nn.init.orthogonal(self.mx.weight.data)
        self.register_weight_drop(self.mx, 'weight', 0.5)
        nn.init.orthogonal(self.mh.weight.data)
        self.register_weight_drop(self.mh, 'weight', 0.5)

    def register_weight_drop(self, module, name, p):
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_raw', nn.Parameter(weight.data))
        self.weight_drop_targets.append((module, name, p))

    def apply_weight_drop(self):
        for module, name, p in self.weight_drop_targets:
            weight = getattr(module, name + '_raw')
            weight = F.dropout(weight, p=p, training=self.training)
            setattr(module, name, weight)

    def forward(self, xs):
        self.apply_weight_drop()

        data, batch_sizes = xs

        h = self.h0.expand(batch_sizes[0], self.hidden_size)
        c = self.c0.expand(batch_sizes[0], self.hidden_size)

        out = []
        last_batch_size = batch_sizes[0]
        off = 0
        for batch_size in batch_sizes:
            x = data[off:off + batch_size]

            if batch_size < last_batch_size:
                out.append(c[batch_size:last_batch_size])
                h = h[:batch_size]
                c = c[:batch_size]

            m = self.mx(x) * self.mh(h)
            h, c = self.cell(x, (m, c))
            off += batch_size
            last_batch_size = batch_size

        out.append(c)
        out.reverse()
        out = torch.cat(out, 0)
        return out


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, out_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim

        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = MLSTM(embedding_dim, out_dim)

    def forward(self, x):
        data, batch_sizes = x
        data = self.embed(data)
        x = PackedSequence(data, batch_sizes)
        x = self.rnn(x)
        return x
