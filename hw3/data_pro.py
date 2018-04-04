import numpy as np
from PIL import Image
import ImageFilter
import matplotlib.pyplot as plt
'''
raw = [row for row in open('train.csv','r')]
trainX = []
trainY = []
OAO = 0
for row in raw[1:]:
	tempy,x = row.split(',')
	y = np.zeros(7)
	y[int(tempy)] = 1
	x = np.array(x.split()).reshape(48,48).astype(np.float)
	img = Image.fromarray(x)
	for ro in range(0,1,4):
		tempx = img.rotate(ro,Image.BILINEAR)

		trainX.append(np.array(tempx))
		trainY.append(y)
	
	
	#print(len(trainX[0].shape))
np.save('trainY' , np.array(trainY))
np.save('trainX' , np.array(trainX))
print(np.array(trainX).shape)
'''
testX = []
raw = [row for row in open('test.csv','r')]
for row in raw[1:]:
	iid , x = row.split(',')
	testX.append(np.array(x.split()).reshape(48,48))
np.save('testX',np.array(testX))
print(np.array(testX).shape)