import numpy as np
import random 
import math
import matplotlib.pyplot as plt

dir_name = 'data/pm25_pm10/'
trainX  = np.load(dir_name + 'train_data.npy')
trainY = np.load(dir_name + 'train_label.npy')
testX  = np.load(dir_name + 'test_data.npy')
N = trainX.shape[0]
print(N)
print(trainX.shape)

one = (trainX - trainX + 1)[:,1].reshape(-1,1)
trainX = np.concatenate( (one,trainX) ,axis=1 )
trainX = np.mat(trainX)

one = (testX - testX + 1)[:,1].reshape(-1,1)
testX = np.concatenate( (one,testX) ,axis=1 )
testX = np.mat(testX)

trainY = np.mat(trainY).transpose()

print(trainX.shape)
print(trainY.shape)
pseudo_inv = np.linalg.pinv(trainX)
w_lin = pseudo_inv * np.transpose(trainY)

print(w_lin)
print( np.sqrt(np.mean(np.square(trainX * w_lin - trainY.T))) )

out = testX * w_lin

outtable = [["id,value"]]


for i in range(len(out)):
	outtable.append(['id_' + str(i),out[i].tolist()[0][0]])


import csv 
cout = csv.writer(open('lin_reg_ans.csv','w'))
cout.writerows(outtable)