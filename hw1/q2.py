# data process
import csv
import numpy as np
np.random.seed(243789891)
data = list(filter(lambda s : 'PM2.5' in s , [ row for row in csv.reader(open('train.csv','r',encoding = 'big5'))]))
data = [np.array(row[3:]).astype(np.float) for row in data]

trainX , trainY = [] , []
for row in data:
	for i in range(len(row)-10):
		r = row[i:i+10]
		if len(r[r<=0]) + len(r[r>120]) > 0:
			continue
		trainX.append(r[:-1])		
		trainY.append([r[-1]])

N = len(trainX)
def GD(w,b):
	w_GD = np.zeros(9)
	b_GD = 0
	rmsq = 0
	for x,y in zip(trainX,trainY):
		n1_loss = np.dot(w,x) + b
		rmsq += np.square(y - n1_loss)
		w_GD += x * ( y - n1_loss )
		b_GD += ( y - n1_loss )
	return -2/N*w_GD , -2/N*b_GD , np.sqrt(rmsq/N)

err_table = [[],[],[],[]]
lr_table = [1.125*1e-4,8*1e-5,4*1e-5,1e-5]
for err , lr in zip(err_table , lr_table):
	print(err,lr)
	w = np.random.normal(0,1,9)
	b = np.random.normal(0,1)

	for ite in range(5000):
		res = GD(w,b)
		w , b = w - lr*res[0] , b - lr*res[1]
		err.append(res[2])
		#print(res[2])
print(err_table)

import matplotlib.pyplot as plt
for err , lr in zip(err_table , lr_table):
	plt.plot(np.log(err) , label='learning rate = %e' %(lr))
plt.legend()
plt.ylabel('log(RMSE)')
plt.xlabel('iteration')
plt.show()