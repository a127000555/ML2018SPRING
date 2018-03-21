# data process
import csv
import numpy as np
np.random.seed(1315142)
#data = list(filter(lambda s : 'PM2.5' in s , [ row for row in csv.reader(open('train.csv','r',encoding = 'big5'))]))
data = list(filter(lambda s : True or 'PM2.5' in s , [ row for row in csv.reader(open('train.csv','r',encoding = 'big5'))]))[1:]
#data = [np.array(row[3:]).astype(np.float) for row in data]
data = [np.array(row[3:]) for row in data]

trainX , trainY = [] , []
ta = len(data[0])
for idx in range(0,len(data),18):
	for time_start in range(ta-10):
		r = []
		for i in range(18):
			factor = data[idx+i][time_start:time_start+9]
			for item in factor:
				item = 0 if item == 'NR' else item
				r.append(item)
		trainX.append(np.array(r).astype(np.float))
		trainY.append([data[idx+9][time_start+10]])

trainX = np.array(trainX).astype(np.float)
trainY = np.array(trainY).astype(np.float)

N = len(trainX)
def GD(w,b):
	n1_loss = np.array(trainY - np.mat(np.dot(trainX,w) + b).T)
	rmsq = np.sqrt(np.sum(np.square(n1_loss))/N)
	w_GD = -2/N*np.sum(trainX  * n1_loss , axis=0)
	b_GD = -2/N*np.sum(n1_loss)
	return w_GD , b_GD , rmsq

w = np.random.normal(0,1,9*18)
b = np.random.normal(0,1)
w = np.array(''' -2.97494566e-01   1.00192120e-02   2.25218254e-01   1.19946612e-01
   1.87164823e-01  -8.04392896e-02   8.30309093e-02  -4.75325322e-01
   1.64874202e-01  -1.10012209e-01   1.12973488e+00  -7.52020915e-01
   1.59199009e+00  -2.36306555e+00   5.30198025e-01   1.35271632e+00
  -6.12370879e-01   1.39848649e+00  -6.50004380e-01   6.15810539e-01
   1.54053968e+00  -8.40106166e-02   4.13678888e-01  -6.70040389e-01
  -6.02343157e-01   3.87869672e-01   1.92972487e+00  -3.01309079e-02
  -1.72055333e-01  -2.24741607e-01  -1.21811445e-01   1.44061298e+00
  -4.30115717e-01  -1.50149355e+00   2.57059385e-01   1.44959407e-01
  -1.49023414e-02  -7.55212846e-01   5.94668143e-02   6.30541967e-01
  -1.83267270e-02   2.02164348e-01  -2.75465102e-02   1.18131569e-01
   1.05506660e-01   2.42923529e-01  -8.61955318e-01  -2.45015839e-01
   3.29284096e-01   8.48760168e-02   1.44811639e-01  -4.52041599e-01
   4.74668066e-01   3.84407709e-01  -1.31512309e-01   8.76525290e-01
  -1.12853343e-01  -4.13990124e-01  -2.53605926e-01  -1.49414868e-01
   1.82147841e-01  -1.89062189e-03  -3.56390621e-02   4.85301845e-02
   1.16123937e-02   5.26951193e-02  -8.28998089e-02  -8.56577427e-02
  -6.62617692e-02  -1.10326539e-01   9.18673173e-02   2.37505029e-01
   2.62348195e-02   5.22973331e-02  -2.56543255e-02   1.67591064e-03
   3.56901262e-02  -6.32017954e-02   2.03866329e-02  -2.42827908e-02
   1.52572805e-01  -1.50822143e-01  -6.33354138e-02  -1.49265320e-02
   2.19915785e-02   2.31066710e-01   2.74913790e-01  -8.31615991e-02
   1.38979252e-01   1.68579049e-01  -1.47556732e-01  -2.35865932e-01
   2.17898389e-01   4.25172628e-01  -6.07758469e-01   2.30852397e-01
   3.24935780e-01  -2.60772177e-01  -6.43112392e-02   2.58896359e-01
  -1.17707478e-01  -1.40040450e-01  -3.04900046e-02   7.68650619e-02
  -1.52559926e-02  -6.63051331e-02   6.41910606e-02  -3.61944862e-02
   1.05585377e+00  -1.38247764e+00   6.19004738e-01  -4.43541383e-01
   3.53242758e-01   2.12786987e-01   7.60775982e-01  -2.78326449e-01
  -7.64822612e-01  -1.46807185e+00   4.02708536e-01  -8.22933285e-02
   6.36027999e-01  -1.86111449e+00  -2.83211931e-01   7.89634906e-02
  -1.00844794e+00  -5.74140702e-01   4.13094452e-03   6.87148987e-04
  -4.31357075e-03   3.06206881e-04   7.63936904e-03   4.36134007e-04
   3.34669344e-03  -3.85044632e-03  -5.47280891e-03  -2.58454097e-03
   3.04312114e-03  -2.48496844e-03  -7.32100416e-04  -7.60271946e-04
   1.62812937e-03   8.12472871e-04   4.90154431e-03   4.62876499e-03
  -3.32361990e-01  -5.64210444e-01  -1.33802536e-01   3.55335233e-01
   3.77595835e-01  -7.23010497e-01   8.97532466e-01  -7.64397448e-01
   4.57494077e-01   4.25724022e-01   9.20676672e-01   1.74208018e+00
  -1.51546875e+00  -5.09695078e-01   4.93966202e-01   1.19933203e-01
   1.38720976e-01  -7.07601536e-01
'''.split()).astype(np.float)
print(w.shape)
lr = 1.0625*1e-6
for ite in range(10):
	res = GD(w,b)
	w -= lr*res[0]
	b -= lr*res[1]
	print(ite,res[2])

#### test process ####
print('{0:-^40s}'.format("output process"))
import sys
#data = list(filter(lambda s : 'PM2.5' in s , [ row for row in csv.reader(open('test.csv','r',encoding = 'big5'))]))
data = list(filter(lambda s : True or 'PM2.5' in s , [ row for row in csv.reader(open('test.csv','r',encoding = 'big5'))]))
outtable = [['id','value']]
'''
for row in data:
	r = np.array(row[2:]).astype(np.float)
	outtable.append( [row[0]  , (np.dot(r,w) + b)])
'''
for idx in range(0,len(data),18):
	r = []
	for i in range(18):
		row = data[idx+i][2:]
		for item in row:
			if item == 'NR':
				r.append(0)
			else:
				r.append(np.float(item))
	r = np.array(r)
	outtable.append( [data[idx][0]  , (np.dot(r,w) + b)])


csv.writer(open('q1_162_params_ans.csv','w')).writerows(outtable)
print('{0:-^40s}'.format("finished"))
#print(w)