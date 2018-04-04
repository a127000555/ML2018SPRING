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

filter_map = ['sex_Female']
A1_augment_filter = ['age','fnlwgt','capital_gain','capital_loss' , 'hours_per_week']
A2_augment_filter = ['sex_Male','race_White','race_Black','relationship_Wife','marital_status_Married-civ-spouse' , 'native_country_Mexico','workclass_Private','workclass_Federal-gov',]

trainX = np.array(df1)
testX = np.array(df2)
for feature in df1.columns.values:
	for times in  np.arange(0.5,4.5,2):
		if feature in A1_augment_filter:
			trainX = np.concatenate( [trainX , (np.array(df1[feature])**times).reshape(-1,1)] , axis=1)
			testX = np.concatenate( [testX , (np.array(df2[feature])**times).reshape(-1,1)] , axis=1)
		if feature in A2_augment_filter:
			trainX = np.concatenate( [trainX , ((np.array(df1[feature])+1)**times).reshape(-1,1)] , axis=1)
			testX = np.concatenate( [testX , ((np.array(df2[feature])+1)**times).reshape(-1,1)] , axis=1)

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
clf = LogisticRegression(penalty = 'l1' , fit_intercept=True, C = 1e15 )

clf.fit(trainX,trainY)
print(clf.score(trainX,trainY))


out = clf.predict(testX)
fout = open('sklearn_log_ans.csv','w')

print('id,label' , file=fout)
_ = 0
for res in out:
	print("%d,%d"%(_+1 , res) , file=fout)
	_+=1

print('finished')
