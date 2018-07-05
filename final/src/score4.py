import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
import re
import jieba
from model4 import Encoder

words = torch.load('./words.pt')
encoder = Encoder(len(words), 256, 512)
encoder.load_state_dict(torch.load('./model4.pt'))
encoder.eval()
encoder.cuda()

jieba.initialize()

pat_split = re.compile('[\t ]')
pat_remove = re.compile('["\-()]')
pat_punc = re.compile('[，、？\.+]')


def clean_text(text):
    text = text.lower()
    text = re.sub(pat_remove, '', text)
    text = re.sub(pat_punc, ' ', text)
    return text


def cut(text):
    return [w for w in jieba.cut(text) if w and w != ' ']


def to_bag(text):
    return [words[w] for w in cut(text) if w in words]


def encode(bags):
    ret = torch.cuda.FloatTensor(encoder.out_dim).zero_()
    for bag in filter(None, bags):
        bag = torch.cuda.LongTensor(bag)
        bag = Variable(bag)
        bag = PackedSequence(bag, batch_sizes=[1] * len(bag))
        ret += encoder(bag).data.squeeze(0)
    return ret


def predict(line):
    line = line.strip()
    _id, x, ys = line.split(',', 2)

    x = re.split(pat_split, x)
    x = map(clean_text, x)
    x = map(to_bag, x)
    x = encode(x)

    ys = ys.split('\t')
    ys = [map(clean_text, y[2:].split(' ')) for y in ys]
    ys = [map(to_bag, y) for y in ys]
    ys = [encode(y) for y in ys]

    scores = [float(torch.dot(x, y)) for y in ys]
    return (_id, scores)


S = []
print('id,ans')
with open('./testing_data.csv', 'r') as f:
    f.readline()
    for _id, scores in map(predict, f.readlines()):
        S.append(scores)
torch.save(S, 'S4.pt')
