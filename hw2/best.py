import pandas
import math
import numpy as np

# Const
sigmoid = lambda s : 1.0 / (1.0 + np.exp(-s))
np.random.seed(777)	
# Hyper Parameters.
lr = 1e-4
df1 = pandas.read_csv('train_X')
df2 = pandas.read_csv('test_X')
trainY = np.array([row for row in open('train_Y','r')]).astype(np.float).reshape(-1,1)

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

quad_test = []
quad_train= []
for i in range(4):
	for j in range(4):
		for k in range(100):
			quad_train.append(np.power(interesting_training_data[i] * interesting_training_data[j],k))
			quad_test.append(np.power(interesting_testing_data[i] * interesting_testing_data[j],k))
quad_test = np.array(quad_test).T
quad_train = np.array(quad_train).T
trainX = np.concatenate( [trainX , quad_train] , axis=1)
testX = np.concatenate( [testX , quad_test] , axis=1)
		
print('quad finished')
for feature in df1.columns.values:
	if feature == 'age' or feature == 'hours_per_week':
		trainX = np.concatenate( [trainX , (np.sinh(np.array(df1[feature]))).reshape(-1,1)] , axis=1)
		testX = np.concatenate( [testX , (np.sinh(np.array(df2[feature]))).reshape(-1,1)] , axis=1)
		trainX = np.concatenate( [trainX , (G(np.array(df1[feature]))).reshape(-1,1)] , axis=1)
		testX = np.concatenate( [testX , (G(np.array(df2[feature]))).reshape(-1,1)] , axis=1)
		#trainX = np.concatenate( [trainX , (inv_G(np.array(df1[feature]))).reshape(-1,1)] , axis=1)
		#testX = np.concatenate( [testX , (inv_G(np.array(df2[feature]))).reshape(-1,1)] , axis=1)
	for times in  np.arange(2,200,2):

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
print(trainX.shape)
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
clf = LogisticRegression(penalty = 'l1' , fit_intercept=True , C =6500 )

print('fitting')
clf.fit(trainX,trainY)
print(clf.score(trainX,trainY))
out = clf.predict(testX)
import pickle
s = pickle.dump(clf,open('best_model','wb'))

fout = open('sklearn_log_ans_OAO.csv','w')
print('id,label' , file=fout)
_ = 0
for res in out:
	print("%d,%d"%(_+1 , res) , file=fout)
	_+=1

print('finished')
