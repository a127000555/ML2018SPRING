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
	w_GD = np.zeros(9)
	b_GD = 0
	rmsq = 0
	for x,y in zip(trainX,trainY):
		n1_loss = np.dot(w,x) + b
		rmsq += np.square(y - n1_loss)
		w_GD += x * ( y - n1_loss )
		b_GD += ( y - n1_loss )
	print("rmsq = %g" %(np.sqrt(rmsq/N)) , end='\n')
	return -2/N*w_GD , -2/N*b_GD
w = np.random.normal(0,1,9)
b = np.random.normal(0,1)
# the vector of 10000 iteration
w=np.array(list(map(np.float,
	'0.02480301 -0.03773     0.22674783 -0.26718364 -0.05165253  0.60141181 -0.64663891 -0.05343139  1.19435038'.split())))
#w = np.array([ 0.00513864 ,0.08738939, 0.02390469,-0.16039784, 0.06559801, 0.35435281,-0.47560475,-0.06156447, 1.16372314]) 
b = np.array([0.27495105])

lr = 1e-4
for ite in range(10):
	res = GD(w,b)
	w -= lr*res[0]
	b -= lr*res[1]

#### test process ####
print('{0:-^40s}'.format("output process"))
import sys
data = list(filter(lambda s : 'PM2.5' in s , [ row for row in csv.reader(open(sys.argv[1],'r',encoding = 'big5'))]))
outtable = [['id','value']]
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
	outtable.append( [row[0]  , (np.dot(r,w) + b)[0]])

csv.writer(open(sys.argv[2],'w')).writerows(outtable)
print('{0:-^40s}'.format("finished"))
