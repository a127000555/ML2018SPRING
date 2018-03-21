# data process
import csv
import numpy as np
np.random.seed(1315142)
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
	n1_loss = np.array(trainY - np.mat(np.dot(trainX,w) + b).T)
	rmsq = np.sqrt(np.sum(np.square(n1_loss))/N)
	w_GD = -2/N*np.sum(trainX  * n1_loss , axis=0)
	b_GD = -2/N*np.sum(n1_loss)
	return w_GD , b_GD , rmsq

w = [np.zeros(9) , np.zeros(9) , np.zeros(9) , np.zeros(9) ] 
b = [np.zeros(1) , np.zeros(1) , np.zeros(1) , np.zeros(1) ]
reg= [1e-1,1e-2,1e-3,1e-4]
lr = 1e-5
for ite in range(20000):
	for idx in range(4):
		res = GD(w[idx],b[idx])
		w[idx] -= lr*res[0] + 2*reg[idx]*(w[idx])
		b[idx] -= lr*res[1]
		if ite %1000== 0:
			print('ite:' , ite,'reg: %.0e'% (reg[idx]) , 'rmsq: ', res[2] , 'w^2 = ' , np.dot(w[idx],w[idx])) 

#### test process ####
print('{0:-^40s}'.format("output process"))
import sys
data = list(filter(lambda s : 'PM2.5' in s , [ row for row in csv.reader(open('test.csv','r',encoding = 'big5'))]))
outtable = [ [['id','value']],[['id','value']],[['id','value']],[['id','value']] ]
for row in data:
	r = np.array(row[2:]).astype(np.float)
	for idx in range(len(r)):
		if len(r[r<=0]) + len(r[r>120]) > 0:
			## data correcting ##
			if idx !=0 and idx != len(r)-1:
				r[idx] = (r[idx-1] + r[idx+1])/2		
			elif idx ==0:
				r[idx] = r[idx+1]
			else:
				r[idx] = r[idx-1]
	for idx in range(4):
		res = [row[0]  , (np.dot(r,w[idx]) + b[idx])[0]]
		outtable[idx].append(res)
for idx in range(4):
	csv.writer(open('q3_reg' + str(reg[idx]) + '_ans.csv','w')).writerows(outtable[idx])
print('{0:-^40s}'.format("finished"))
