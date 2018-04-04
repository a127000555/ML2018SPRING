import pandas
import math
import numpy as np
import random
# Const
sigmoid = lambda s : 1.0 / (1.0 + np.exp(-s))
np.random.seed(777)	
# Hyper Parameters.
lr = 1e-4
df = pandas.read_csv('train_X')

trainY = np.array([row for row in open('train_Y','r')]).astype(np.float)
feature= ['age','sex_Female']
trainX = np.array(df)
idx = list(range(len(trainX)))
testX = np.array(pandas.read_csv('test_X'))
# trainX = np.array(df[feature])
# testX = np.array(pandas.read_csv('test_X')[feature])
best_w , best_b ,best_score = None , None , 0
def train(trainX,trainY,testX,testY):
	global best_w,best_b,best_score
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
	score = 1- np.count_nonzero(np.square(y - testY))/len(testX)
	if best_score < score:
		best_w , best_b , best_score = w , b , score
		print( '**' , score  , '**')
		
		out = 1-(sigmoid(np.matmul(testX,best_w)+best_b)>0.5).astype(np.int)
		fout = open('prob_ans.csv','w')
		print('id,label',file=fout)
		for _ in range(len(testX)):
			print("%d,%d"%(_+1,out[_]) , file=fout)
	else:
		print( score )
		

for sp in range(3000):
	random.shuffle(idx)
	split = int(len(trainX)*np.random.uniform(0.7,0.8))
	drop = int(len(trainX)*np.random.uniform(0.9,1))
	#drop = 0.1
	print('epoch ',sp,end=' ')
	train(trainX[idx[:split]],trainY[idx[:split]],trainX[idx[split:drop]],trainY[idx[split:drop]])


out = 1-(sigmoid(np.matmul(testX,best_w)+best_b)>0.5).astype(np.int)
fout = open('prob_ans.csv','w')
print('id,label',file=fout)
for _ in range(len(testX)):
	print("%d,%d"%(_+1,out[_]) , file=fout)