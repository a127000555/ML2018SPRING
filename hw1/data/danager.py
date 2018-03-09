import numpy as np

import pickle
def generate():
	train_in = pickle.load(open('date_process/train.pickle','rb'))
	test_in = pickle.load(open('date_process/test.pickle','rb'))

	train_data = []
	train_label = []
	test_data = []

	### param ###
	time_range = 5
	#############


	## training prepare
	for day in train_in + test_in:
		pm25 = np.array(day['PM2.5']).astype(np.float)
		pm10 = np.array(day['PM10']).astype(np.float)
		pm25_mean = np.mean(pm25)
		pm10_mean = np.mean(pm10)
		N = len(pm25)
		for i in range(N-time_range):
			data = [pm25_mean] + list(pm25[i:i+time_range]) + list(pm10[i:i+time_range])
			y = [pm25[i+time_range]]
			train_data.append(data)
			train_label.append(y)


	## testing process
	for day in test_in:
		pm25 = np.array(day['PM2.5']).astype(np.float)
		pm10 = np.array(day['PM10']).astype(np.float)
		pm25_mean = np.mean(pm25)
		pm10_mean = np.mean(pm10)
		N = len(pm25)
		data = [pm25_mean] + list(pm25[-time_range:]) + list(pm10[-time_range:])
		test_data.append(data)

	dir_name = 'pm25_pm10'
	np.save(dir_name + '/' + 'train_data' , train_data)
	np.save(dir_name + '/' + 'train_label' , train_label)
	np.save(dir_name + '/' + 'test_data' , test_data)

	for _ in range(10):
		print(train_data[_])

def split(dir_name,ratio=0.2):
	trainX  = np.load(dir_name + '/train_data.npy')
	trainY = np.load(dir_name + '/train_label.npy')
	N = trainX.shape[0]
	split_point = int(N*(1-ratio))
	import random
	rand_idx = list(range(N))
	random.shuffle(rand_idx)
	tempX, tempY = [] , []
	
	for idx in rand_idx:
		tempX.append(trainX[idx])
		tempY.append(trainY[idx])
	
	trainX = np.array(tempX)	
	trainY = np.array(tempY)
	return trainX[:split_point ] , trainY[:split_point ] , trainX[split_point :],trainY[split_point :]
