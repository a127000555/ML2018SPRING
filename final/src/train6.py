import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime
from torch.nn.utils.rnn import PackedSequence
import random

from model6 import Encoder

WINDOW_SIZE = 3
N_CHOICE = 6

TOTAL_EPOCH = 300
BATCH_SIZE = 512

X = torch.load('./X.pt')
words = torch.load('./words.pt')

N = len(X)
N_BATCH = N // BATCH_SIZE

encoder = Encoder(len(words), 256, 512)
encoder.embed.weight.data.copy_(torch.load('./weight.pt'))
encoder.cuda()

optimizer = torch.optim.Adam(encoder.parameters(), lr=5e-4)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda t: 1 - t / TOTAL_EPOCH,
)

criterion = nn.CrossEntropyLoss()


def sort_by_length(xs):
    keys = sorted(range(len(xs)), key=lambda i: len(xs[i]), reverse=True)
    xs = [xs[i] for i in keys]
    return xs, keys


def inverse_perm(keys):
    inv = torch.LongTensor(len(keys))
    for i, k in enumerate(keys):
        inv[k] = i
    return inv


def pad(xs):
    buf = torch.LongTensor(len(xs[0]), len(xs)).zero_()
    for i, x in enumerate(xs):
        buf[:len(x), i] = x
    ls = [len(x) for x in xs]
    return buf, ls


def pack(xs):
    xs, ls = pad(xs)
    xs = nn.utils.rnn.pack_padded_sequence(xs, ls)
    xs = PackedSequence(Variable(xs.data.cuda()), xs.batch_sizes)
    return xs


def train(xys, inv):
    optimizer.zero_grad()

    xys = encoder(xys)
    xys = xys.index_select(0, inv)

    xs = xys[:BATCH_SIZE].view(BATCH_SIZE, 1, encoder.out_dim)
    ys = xys[BATCH_SIZE:].view(BATCH_SIZE, N_CHOICE, encoder.out_dim)

    scores = xs * ys
    scores = nn.functional.dropout(scores, 0.5, training=True)
    scores = scores.sum(dim=2)

    ans = torch.cuda.LongTensor(BATCH_SIZE).zero_()
    loss = criterion(scores, Variable(ans))
    loss.backward()
    optimizer.step()

    return loss.data[0]


print('start training')

for t in range(TOTAL_EPOCH):
    start_time = datetime.now()
    scheduler.step()

    perm = torch.randperm(N - WINDOW_SIZE)

    loss_sum = 0
    for i in range(N_BATCH):
        p = perm.narrow(0, i * BATCH_SIZE, BATCH_SIZE)
        xs = [X[i] for i in p]
        ys_correct = [X[i + random.randint(1, WINDOW_SIZE)] for i in p]

        ys = []
        for i in range(BATCH_SIZE):
            ys.append(ys_correct[i])
            ys += random.sample(X, N_CHOICE - 1)

        xys, keys = sort_by_length(xs + ys)
        inv = inverse_perm(keys)
        inv = Variable(inv.cuda())
        xys = pack(xys)
        loss_sum += train(xys, inv)

    print('%d %.6f %s' % (t + 1, loss_sum / N_BATCH,
                          datetime.now() - start_time))

    torch.save(encoder.state_dict(), 'model6.pt')
