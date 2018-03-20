import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
train = pickle.load(open('date_process/train.pickle','rb'))

file_list = ['train.csv']
df = pd.read_csv(file_list[0],encoding='big5')
print(type(df))


import csv 
cin = csv.reader(open(file_list[0],'r',encoding='big5'))
# RAIN_FALL
all_factor=['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR','RAINFALL','PM2.5','WD_SIN','WD_COS']
factor_enumerate = []
for i in range(len(all_factor)):
	for j in range(i+1 , len(all_factor)):
		factor_enumerate.append ( (all_factor[i] , all_factor[j]) )
raw = [row for row in cin]


for factor in factor_enumerate:
	#factor = ['WIND_SPEED','PM2.5']
	time_st = 0
	param1 = []
	param2 = []
	for day in train:
		param1.append(day[factor[0]])
		param2.append(day[factor[1]])
		#if time_st  > 400:
		#	break
	data = []
	for X,Y in zip(param1 , param2):
		for x,y in zip(X,Y):
			if x != factor[0]:
				data.append([x,y])
	data.sort()
	data = np.array(data).astype(np.float)
	#print(data.tolist())
	import matplotlib.pyplot as plt
	plt.xlabel(factor[0])
	plt.ylabel(factor[1])
	try:
		x_data = data[:,0]
		y_data = data[:,1]

		corrcoef = stats.pearsonr(x_data,y_data)
		
		#plt.xlim(-5,7.5)
		plt.legend()
		plt.title(corrcoef[0])
		plt.scatter(x = x_data ,y = y_data,marker='.')
		import os
		dir_name = 'feature_visualize/two_factor_observation'
		png_name =  factor[0] + '_' + factor[1] + '.png'
		
		try:
			os.mkdir(dir_name + '/' + factor[0])
		except:
			pass
		try:
			os.mkdir(dir_name + '/' + factor[1])
		except:
			pass
		try:
			os.mkdir(dir_name + '/highly_corrcoef')
		except:
			pass
		
		if abs(corrcoef[0]) > 0.4 :
			plt.savefig(dir_name + '/highly_corrcoef/' + png_name)
		
			print(factor , corrcoef)
		
		plt.savefig(dir_name + png_name)
		plt.savefig(dir_name + '/' + factor[0] + '/' + png_name)
		plt.savefig(dir_name + '/' + factor[1] + '/' + png_name)
		plt.clf() 
	except Exception as e:
		print(factor , end = '')
		print('Error:' , e)
