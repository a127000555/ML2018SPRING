import numpy as np
import random 
import math
import matplotlib.pyplot as plt


immune = set()
trash = []
trash_y = []
def lin_reg(trainX , trainY):
	trainX = np.mat(trainX)
	trainY = np.mat(trainY)
	#print(trainX.shape , trainY.shape)

	'''
	one = (trainX - trainX + 1)[:,1].reshape(-1,1)
	trainX = np.concatenate( (one,trainX) ,axis=1 )
	trainX = np.mat(trainX)
	'''
	

	pseudo_inv = np.linalg.pinv(trainX)
	w_lin = pseudo_inv * trainY

	#print(w_lin.transpose())
	#print( (trainX * w_lin).shape , trainY.shape)
	rmse = np.sqrt(np.mean(np.square(trainX * w_lin - trainY)))
	
	error_table = []
	y_ = trainY.tolist()
	x = trainX.tolist()
	y = (trainX * w_lin).transpose().tolist()[0]

	for _ in range(len(y)):
		error_table.append(
				(
					(y_[_][0] - y[_])**2,
					y_[_],
					y[_],
					x[_],
					_
				)
			)
	sort_table = sorted(error_table, reverse=True)
	for i in range(min(0,len(immune)), len(sort_table)):
		if str( (sort_table[i][3] , sort_table[i][1][0]) ) not in immune:
			#print('OAO:',str( (sort_table[i][3] , sort_table[i][1][0]) ))
			return {
					
					"w_lin" : w_lin,
					"error_id" : sort_table[i][4],
					"error_info" : sort_table[i],
					"RMSE" : rmse
				}
def output(w_lin , testX):
	'''
	one = (testX - testX + 1)[:,1].reshape(-1,1)
	testX = np.concatenate( (one,testX) ,axis=1 )
	testX = np.mat(testX)
	'''
	out = testX * w_lin
	outtable = [['id','value']]

	
	for i in range(len(out)):
		outtable.append(['id_' + str(i),out[i].tolist()[0][0]])


	import csv 
	cout = csv.writer(open('lin_reg_ans2.csv','w'))
	cout.writerows(outtable)
	print('OAO')

if __name__ == '__main__':
	dir_name = 'data/pm25_pm10(time_cat)/'
	#dir_name = 'data/pm25_pm10/'
	trainX  = 	np.load(dir_name + 'train_data.npy').tolist()
	trainY = 	np.load(dir_name + 'train_label.npy').tolist()
	testX  = 	np.load(dir_name + 'test_data.npy')
	''''
	for row in testX:
		for element in row:
			print("%5g" %( element) , end=' ')
		print()
	exit(0)
	'''
	OldX = trainX
	OldY = trainY
	w_lin = lin_reg(trainX,trainY)["w_lin"]
	for _ in range(2000):
		res = lin_reg(trainX,trainY)
		w_lin = res["w_lin"]
		info = res["error_info"]
		print('RMSE:' , res["RMSE"] , 'SE:' , info[0] ,'\ttrue_y' , info[1][0]  , '\tpred_y' , info[2] , '\tgiven_x' , info[3])
		given_x = info[3]
		pred_y 	= info[2]
		given_y = info[1][0]
		if (min(given_x) - 5 <= float(given_y) and float(given_y) <= max(given_x) + 5 ) or abs(given_y - given_x[-2] + 2 * given_x[-1]) < 2 :
			immune.add(  str( (given_x , given_y)) )
			print("resonable.\t",w_lin.T)
		else:
			print("irresonable.\t",w_lin.T)
			trash.append(given_x)
			trash_y.append(given_y)
			del trainX[res["error_id"]]
			del trainY[res["error_id"]]
		#print(trainX.shape)
		#print(immune)
	print("{0:-^100s}".format("immune-list"))
	for _ in immune:
		print(_)
	print("{0:-^100s}".format("trash-list"))
	for _ in range(len(trash)):
		print("x:" , trash[_] , "\ty:" , trash_y[_])
	print(w_lin.transpose())
	output(w_lin , testX)
