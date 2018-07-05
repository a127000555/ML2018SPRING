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
S5 = torch.load('S4.pt')

S = [[a+b for a, b in zip(s4, s5)] for s4, s5 in zip(S4, S5)]

print('id,ans')
for _id, s in enumerate(S):
    ks = range(6)
    if _id in patch:
        ks = [patch[_id]]
    elif _id in reject:
        k = [k for k in ks if k not in reject[_id]]
    ans = max(ks, key=lambda i: s[i])
    print('%d,%s' % (_id, ans))
