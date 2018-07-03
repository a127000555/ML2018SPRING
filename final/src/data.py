from pathlib import Path
import re
import jieba
import torch
import numpy as np
from gensim.models.word2vec import Word2Vec

SIZE = 256

pat_remove = re.compile('["\-()]')
pat_punc = re.compile('[，、？\.+]')


def clean_text(text):
    text = text.lower()
    text = re.sub(pat_remove, '', text)
    text = re.sub(pat_punc, ' ', text)
    return text


def cut(text):
    return [w for w in jieba.cut(text) if w and w != ' ']


def to_bag(ws, words):
    return torch.LongTensor([words[w] for w in ws])


jieba.initialize()

X = []
words = set()
for p in Path('./training_data').iterdir():
    doc = p.read_text()
    doc = clean_text(doc)
    lines = doc.split('\n')
    lines = map(cut, lines)
    lines = list(filter(None, lines))

    X += lines
    for line in lines:
        words.update(line)

words = {w: i for i, w in enumerate(words)}
torch.save([to_bag(x, words) for x in X], './X.pt')
torch.save(words, './words.pt')

print('start training')

model = Word2Vec(
    X,
    sg=1,
    size=SIZE,
    min_count=1,
    alpha=0.1,
    iter=100,
    workers=12,
)

weight = np.empty([len(words), SIZE], np.float32)
for w in words:
    weight[words[w], :] = model[w]
torch.save(torch.FloatTensor(weight), './weight.pt')
