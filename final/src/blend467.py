import torch

patch = dict()
reject = dict()

with open('./testing_data.csv', 'r') as f:
    f.readline()
    for line in f.readlines():
        line = line.strip()
        _id, x, ys = line.split(',', 2)
        _id = int(_id)
        ys = ys.split('\t')
        ys = [int(y[0]) for y in ys if '"' in y]

        if '"' in x:
            if len(ys) == 1:
                patch[_id] = ys[0]
        elif len(ys) > 0:
            reject[_id] = ys

S4 = torch.load('S4.pt')
S6 = torch.load('S6.pt')
S7 = torch.load('S7.pt')

S = [[a + b / 2 + c / 2 for a, b, c in zip(*t)] for t in zip(S4, S6, S7)]

print('id,ans')
for _id, s in enumerate(S):
    ks = range(6)
    if _id in patch:
        ks = [patch[_id]]
    elif _id in reject:
        ks = [k for k in ks if k not in reject[_id]]
    ans = max(ks, key=lambda i: s[i])
    print('%d,%s' % (_id, ans))
