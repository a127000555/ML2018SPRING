import pandas
import math
import numpy as np


import sys

# file name #
train_X_name = 'train_X'
train_Y_name = 'train_Y'
test_X_name = 'test_X'
test_Y_name = 'test_Y'
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
df2 = pandas.read_csv(test_X_name)
trainY = np.array([row for row in open(train_Y_name,'r')]).astype(np.float).reshape(-1,1)

A1_augment_filter = ['age','capital_gain','capital_loss' , 'hours_per_week']

trainX = np.array(df1)
testX = np.array(df2)
def G(x):
	return np.exp(-np.power(x,2))
def inv_G(x):
	return 1/np.exp(-np.power(x,2))

interesting_training_data = []
interesting_testing_data = []

for feature in A1_augment_filter:
	interesting_training_data.append(df1[feature]) 
	interesting_testing_data.append(df2[feature])

interesting_testing_data = np.array(interesting_testing_data)
interesting_training_data = np.array(interesting_training_data)
for feature in df1.columns.values:
	if feature == 'age' or feature == 'hours_per_week':
		trainX = np.concatenate( [trainX , (np.sinh(np.array(df1[feature]))).reshape(-1,1)] , axis=1)
		testX = np.concatenate( [testX , (np.sinh(np.array(df2[feature]))).reshape(-1,1)] , axis=1)
		trainX = np.concatenate( [trainX , (G(np.array(df1[feature]))).reshape(-1,1)] , axis=1)
		testX = np.concatenate( [testX , (G(np.array(df2[feature]))).reshape(-1,1)] , axis=1)
		#trainX = np.concatenate( [trainX , (inv_G(np.array(df1[feature]))).reshape(-1,1)] , axis=1)
		#testX = np.concatenate( [testX , (inv_G(np.array(df2[feature]))).reshape(-1,1)] , axis=1)
	for times in  np.arange(2,100,3):

		if feature in A1_augment_filter:
			trainX = np.concatenate( [trainX , (np.power(np.array(df1[feature]),times)).reshape(-1,1)] , axis=1)
			testX = np.concatenate( [testX , (np.power(np.array(df2[feature]),times)).reshape(-1,1)] , axis=1)


	if feature == 'age' or feature == 'hours_per_week':
		trainX = np.concatenate( [trainX , (np.power(1.1,np.array(df1[feature]/50))).reshape(-1,1)] , axis=1)
		testX = np.concatenate( [testX , (np.power(1.1,np.array(df2[feature])/50)).reshape(-1,1)] , axis=1)
		trainX = np.concatenate( [trainX , (np.power(1.5,np.array(df1[feature]/50))).reshape(-1,1)] , axis=1)
		testX = np.concatenate( [testX , (np.power(1.5,np.array(df2[feature])/50)).reshape(-1,1)] , axis=1)
	if feature == 'fnlwgt':
		for times in  np.arange(2,5,1.2):
			trainX = np.concatenate( [trainX , (np.power(np.array(df1[feature]),times)).reshape(-1,1)] , axis=1)
			testX = np.concatenate( [testX , (np.power(np.array(df2[feature]),times)).reshape(-1,1)] , axis=1)

## append bias 
X = np.concatenate([trainX,testX],axis=0)
Xmean = np.mean(X,axis=0)
Xstd = np.std(X,axis=0)
Xstd = [ 1e-10 if Xstd[i]==0 else Xstd[i] for i in range(len(Xstd))]
trainX = (trainX - Xmean) / Xstd
testX = (testX - Xmean) / Xstd
trainX = np.concatenate( (trainX,np.ones([trainX.shape[0],1])),axis=1)
testX = np.concatenate( (testX,np.ones([testX.shape[0],1])),axis=1)


from sklearn.linear_model import LogisticRegression
import pickle

clf = pickle.load(open('best_model','rb'))
out = clf.predict(testX)

fout = open(test_Y_name,'w')
print('id,label' , file=fout)
_ = 0
for res in out:
	print("%d,%d"%(_+1 , res) , file=fout)
	_+=1

print('finished')
