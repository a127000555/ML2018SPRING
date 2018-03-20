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
	concentrate = ['PM2.5' , 'PM10' ]#, 'CO']
	for day in train_in + test_in :
		pm25 = np.array(day['PM2.5']).astype(np.float)
		pm10 = np.array(day['PM10']).astype(np.float)
		co = np.array(day['CO']).astype(np.float)
		pm25_mean = np.mean(pm25)
		pm10_mean = np.mean(pm10)
		co_mean = np.mean(co)

		N = len(pm25)
		for i in range(N-time_range):
			data =  [pm25_mean , pm10_mean , co_mean] + list(pm25[i:i+time_range]) + list(pm10[i:i+time_range])  + list(co[i:i+time_range])
			#data =  list(pm25[i:i+time_range]) + list(pm10[i:i+time_range])
			y = [pm25[i+time_range]]
			train_data.append(data)
			train_label.append(y)
	
	# concatenate
	'''
	for day_idx in range(0 , len(train_in)-1 , 2 ):
		id1 = train_in[day_idx]['id'] 
		id2 = train_in[day_idx+1]['id']
		if int(id1.split('/')[-1]) +1 != int(id2.split('/')[-1]):
			continue
		print('concatenate : ' , id1 , id2 )
		pm25 = np.array(train_in[day_idx]['PM2.5'][-time_range+1:] + train_in[day_idx]['PM2.5'][:time_range-1]).astype(np.float)
		pm10 = np.array(train_in[day_idx]['PM10'][-time_range+1:] + train_in[day_idx]['PM10'][:time_range-1]).astype(np.float)
		co = np.array(train_in[day_idx]['CO'][-time_range+1:] + train_in[day_idx]['CO'][:time_range-1]).astype(np.float)
		pm25_mean = np.mean(pm25)
		pm10_mean = np.mean(pm10)
		co_mean = np.mean(co)

		N = len(pm25)
		for i in range(N-time_range):
			#data =  [pm25_mean , pm10_mean , co_mean] + list(pm25[i:i+time_range]) + list(pm10[i:i+time_range])  + list(co[i:i+time_range])
			data =  [pm25_mean , pm10_mean] + list(pm25[i:i+time_range]) + list(pm10[i:i+time_range]) 
			y = [pm25[i+time_range]]
			train_data.append(data)
			train_label.append(y)
		'''
	## testing process
	for day in test_in:
		pm25 = np.array(day['PM2.5']).astype(np.float)
		pm10 = np.array(day['PM10']).astype(np.float)
		pm10 = np.array(day['CO']).astype(np.float)
		pm25_mean = np.mean(pm25)
		pm10_mean = np.mean(pm10)
		co_mean = np.mean(co)
		N = len(pm25)
		data = [pm25_mean , pm10_mean,co_mean ] + list(pm25[-time_range:]) + list(pm10[-time_range:]) + list(co[-time_range:])
		#data =  list(pm25[-time_range:]) + list(pm10[-time_range:])
		print(data)
		test_data.append(data)


	dir_name = 'pm25_pm10'
	np.save(dir_name + '/' + 'train_data' , (np.array(train_data)))
	np.save(dir_name + '/' + 'train_label' ,  (np.array(train_label)))
	np.save(dir_name + '/' + 'test_data' ,  (np.array(test_data)))
	print(np.array(train_data).shape)
	print(np.array(train_label).shape)
	print(np.array(test_data).shape)

	#for _ in range(10):
	#	print(train_data[_])

if __name__ == '__main__':
	generate()

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

#generate()