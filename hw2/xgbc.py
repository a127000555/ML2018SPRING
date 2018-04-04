from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import random
import pandas
import warnings
random.seed(100)

xgbc = XGBClassifier()
train_X = np.array( pd.read_csv('train_X'))
test    = np.array([row[:-1].split(',') for row in open('test_X' ,'r')][1:])
train_Y = np.array([i[0] for i in open('train_Y','r')])
N = len(train_X)
idx = [ i for i in range(N) ]
now_best = 0
random.shuffle(idx)
split = int(N*0.3)
batch_X = train_X[idx[:split]]
batch_Y = train_Y[idx[:split]]
valid_X = train_X[idx[split:]]
valid_Y = train_Y[idx[split:]]

xgbc.fit(batch_X,batch_Y)
score = xgbc.score(valid_X,valid_Y)
print(xgbc.get_params())
print(score)
if now_best < score:
	out = xgbc.predict(test)
	fout = open('ans.csv','w')
	print('id,label',file=fout)
	[print('%d,%d'%(i+1,out[i]),file=fout) for i in range(len(out))]
