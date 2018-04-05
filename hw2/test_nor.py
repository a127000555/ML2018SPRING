import pandas
import math
import numpy as np


import sys

# file name #
train_X_name = 'train_X'
train_Y_name = 'train_Y'
test_X_name = 'test_X'
test_Y_name = 'test_Y_no_nor'
if len(sys.argv) != 1:
	train_X_name = sys.argv[3]
	train_Y_name = sys.argv[4]
	test_X_name = sys.argv[5]
	test_Y_name = sys.argv[6]

# Const
sigmoid = lambda s : 1.0 / (1.0 + np.exp(-s))
np.random.seed(777)	
# Hyper Parameters.
lr = 1e-4
df1 = pandas.read_csv(train_X_name)
trainX = pandas.read_csv(train_X_name)
testX = pandas.read_csv(test_X_name)
trainY = np.array([row for row in open(train_Y_name,'r')]).astype(np.float).reshape(-1,1)

## append bias 
X = np.concatenate([trainX,testX],axis=0)

datamin = np.min(np.concatenate( (trainX,testX),axis=0) , axis=0)
datamax = np.max(np.concatenate( (trainX,testX),axis=0) , axis=0)
#trainX = (trainX - datamin) / (datamax - datamin)
#testX = (testX - datamin) / (datamax - datamin)
trainX = np.concatenate( (trainX,np.ones([trainX.shape[0],1])),axis=1)
testX = np.concatenate( (testX,np.ones([testX.shape[0],1])),axis=1)

def cal(x,w):
	return sigmoid(np.dot(x , w))
w = np.zeros([trainX.shape[1]])

lr= 1e-3

def gradient_descent(X,Y,w):
	y = sigmoid(np.dot(X , w))
	w_grad =  np.mean( -X.T *(Y.reshape(-1)-y),axis=1)
	return w_grad


for epoch in range(2000):
	w = w -  lr*gradient_descent(trainX, trainY , w) 
	if epoch % 100 == 99:
		print('epoch' , epoch , ':', 1- np.count_nonzero(trainY.reshape(-1) - (cal(trainX,w)>0.5).astype(int))/len(trainX))
		#print(np.linalg.norm(w))
		#print(w)
out = (cal(testX,w)>0.5).tolist()
fout = open(test_Y_name,'w')
print('id,label' , file=fout)
for _,res in zip(range(len(out)),out):
	print("%d,%d"%(_+1 , res) , file=fout)

for importency,feature in zip(w,df1.columns.values):
	print(feature , importency)
