import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils import weight_norm


class MLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.mx = nn.Linear(input_size, hidden_size)
        self.mh = nn.Linear(hidden_size, hidden_size)

        self.h0 = nn.Parameter(torch.zeros(hidden_size))
        self.c0 = nn.Parameter(torch.zeros(hidden_size))

        cell_weights = []
        for name, param in self.cell.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal(param.data)
                cell_weights.append(name)
            elif 'bias' in name:
                param.data[hidden_size:hidden_size * 2] = 1

        for name in cell_weights:
            weight_norm(self.cell, name)

        nn.init.orthogonal(self.mx.weight.data)
        weight_norm(self.mx, 'weight')
        nn.init.orthogonal(self.mh.weight.data)
        weight_norm(self.mh, 'weight')

    def forward(self, xs):
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
        super(Encoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim

        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.drop = nn.Dropout(0.5)
        self.rnn = MLSTM(embedding_dim, out_dim)

    def forward(self, x):
        data, batch_sizes = x
        data = self.embed(data)
        data = self.drop(data)
        x = PackedSequence(data, batch_sizes)
        x = self.rnn(x)
        return x
