import numpy as np
import random 
import math
import matplotlib.pyplot as plt


dir_name = 'pm25_pm10(time_cat)/'
trainX  = 	np.load(dir_name + 'train_data.npy')
trainY = 	np.load(dir_name + 'train_label.npy')
buc= []
for x,y in zip(trainX,trainY):
	if 2 < y[0] < 120:
		plt.scatter(x=np.mean(x),y=y[0],s=1 , color='blue')
	buc.append(np.mean(x))

buc.sort()
print(buc)
print(buc[len(buc)//2])
plt.show()