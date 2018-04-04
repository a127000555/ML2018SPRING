import pandas
import math
import numpy as np
import random
import sys

# file name #
train_X_name = 'train_X'
train_Y_name = 'train_Y'
test_X_name = 'test_X'
test_Y_name = 'test_Y'
if len(sys.argv) != 2:
	train_X_name = sys.argv[3]
	train_Y_name = sys.argv[4]
	test_X_name = sys.argv[5]
	test_Y_name = sys.argv[6]
  

# Const
sigmoid = lambda s : 1.0 / (1.0 + np.exp(-s))
np.random.seed(777)	
# Hyper Parameters.
lr = 1e-4
df = pandas.read_csv(train_X_name)

trainY = np.array([row for row in open(train_Y_name,'r')]).astype(np.float)
feature= ['age','sex_Female']
trainX = np.array(df)
idx = list(range(len(trainX)))
testX = np.array(pandas.read_csv(test_X_name))
# trainX = np.array(df[feature])
# testX = np.array(pandas.read_csv('test_X')[feature])
best_w , best_b ,best_score = None , None , 0

cla0 = np.argwhere(trainY==0)[:,0]
cla1 = np.argwhere(trainY==1)[:,0]
mean0= np.mean(trainX[cla0].T,axis=1)
mean1= np.mean(trainX[cla1].T,axis=1)
N0 = cla0.shape[0]
N1 = cla1.shape[0]
cov0= np.cov(trainX[cla0].T)
cov1= np.cov(trainX[cla1].T)
icov = np.linalg.pinv((N0*cov0 + N1*cov1)/(N0+N1))
w = np.matmul((mean0-mean1).T,icov)
b = - .5 * np.matmul(np.matmul(mean0.T,icov),mean0) \
	+ .5 * np.matmul(np.matmul(mean1.T,icov),mean1) +  np.log(N0/N1) 
y = 1-(sigmoid(np.matmul(testX,w)+b)>0.5).astype(np.int)

score =  (1-(sigmoid(np.matmul(trainX,w)+b)>0.5).astype(np.int)) != trainY

out = 1-(sigmoid(np.matmul(testX,w)+b)>0.5).astype(np.int)
fout = open(test_Y_name,'w')
print('id,label',file=fout)
for _ in range(len(testX)):
	print("%d,%d"%(_+1,out[_]) , file=fout)

