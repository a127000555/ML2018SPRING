import csv
import numpy as np 
cin = csv.reader(open('train.csv','r',encoding='big5'))
table = [row for row in cin]
#print(np.array(table))
stamp = 0

banned = ['2014/3/12','2014/4/2' , '2014/4/6']

all_training = []
for row_idx_head in range(1,len(table),18):
	if table[row_idx_head][0] in banned:
		continue
	day = { 'id' : table[row_idx_head][0] }
	for row_idx in range(row_idx_head,row_idx_head+18):
		row = table[row_idx]
		day.update({ row[2] : row[3:]})
	# RAINFALL concert
	rain = day['RAINFALL']
	for idx in range(len(rain)):
		if rain[idx] == 'NR' or float(rain[idx]) < 3:
			rain[idx]  = '0.0'
		else:
			rain[idx] = '1'
	day.update( {'RAINFALL' : rain})

	# PM2.5 error correct
	pm = list(map(float,day['PM2.5']))
	try:
		for idx in range(len(pm)):
			if pm[idx] < 1 or pm[idx] > 110:
				raise Error()

			if 0 < idx and idx < len(pm)-1:
				if abs (pm[idx-1] - pm[idx+1] ) < 20 and abs (pm[idx] - pm[idx+1]) > 18:
					pass
					#print((pm[idx-1],pm[idx],pm[idx+1]),pm , sep='\n') 
					#raise Error()
	except:
		continue

	day.update( {'PM2.5' : pm})
	# PM10 error correct
	pm = list(map(float,day['PM10']))
	try:
		for item in pm:
			if item == 0 or item >200:
				#print(pm)
				raise Error()
	except:
		continue
	
	day.update( {'PM10' : pm})


	# CO error correct
	co = list(map(float,day['CO']))
	try:
		pass	
		#print(np.array(co))	
		# for item in co:
		# 	if item == 0 or item >200:
				# print(pm)
				# raise Error()
	except:
		continue
	
	day.update( {'CO' : co})

	wd = list(map(float,day['WIND_DIREC']))
	try:
		day.update( {'WD_COS' : np.cos(np.array(wd) * np.pi / 180)})
		day.update( {'WD_SIN' : np.sin(np.array(wd) * np.pi / 180)})
		print(np.cos(np.array(wd) *np.pi / 180))
		print(np.sin(np.array(wd) *np.pi / 180))
		# for item in co:
		# 	if item == 0 or item >200:
				# print(pm)
				# raise Error()
	except Exception as e:
		print(e)
		exit(0)
		continue

	

	all_training.append(day)
	
	#exit(0)
#print(all_training)

import pickle
with open('date_process/train.pickle','wb') as fout:
	pickle.dump(all_training,fout)		

cin = csv.reader(open('test.csv','r',encoding='big5'))
table = [row for row in cin]
#print(np.array(table) , len(table),18)

all_training = []
for row_idx_head in range(0,len(table),18):
	if table[row_idx_head][0] in banned:
		continue
	day = { 'id' : table[row_idx_head][0] }
	for row_idx in range(row_idx_head,row_idx_head+18):
		row = table[row_idx]
		day.update({ row[1] : row[2:]})

	# RAINFALL correct
	rain = day['RAINFALL']
	for idx in range(len(rain)):
		if rain[idx] == 'NR':
			rain[idx]  = '0.0'
	day.update( {'RAINFALL' : rain})
	

	# PM2.5 error correct
	pm = list(map(float,day['PM2.5']))
	try:
		for idx in range(len(pm)):
			if pm[idx] <= 0 :
				if 0 < idx and idx < len(pm)-1 and 5 < pm[idx+1] and pm[idx+1]< 100:
					pm[idx] = (pm[idx-1] + pm[idx+1])/2
				else:
					# edge processing
					if idx == 0:
						pm[idx] = ( pm[idx+1] + pm[idx+2] )/2
					if idx == len(pm)-1:
						pm[idx] = pm[idx-1]
	except:
		continue
	try:
		for idx in range(len(pm)):
			if 0 < idx and idx < len(pm)-1:
				if abs (pm[idx-1] - pm[idx+1] ) < 20 and abs (pm[idx] - pm[idx+1]) > 20:
					pm[idx] = ( pm[idx-1] + pm[idx+1] )/2
					#print((pm[idx-1],pm[idx],pm[idx+1]),pm , sep='\n') 
	except:
		continue

	


	day.update( {'PM2.5' : pm})
	#print(day['PM2.5'])


	# PM10 error correct
	pm = list(map(float,day['PM10']))
	try:
		for idx in range(len(pm)):
			if pm[idx] <= 0 :

				if 0 < idx and idx < len(pm)-1 and 5 < pm[idx+1] and pm[idx+1]< 100:
					#print('=>' , pm)
					pm[idx] = (pm[idx-1] + pm[idx+1])/2
					#print('<=' , pm)
					
				else:
					# edge processing
					if idx == 0:
						pm[idx] = ( pm[idx+1] + pm[idx+2] )/2
					if idx == len(pm)-1:
						pm[idx] = pm[idx-1]
	except:
		continue
		


	day.update( {'PM10' : pm})
	all_training.append(day)

#print(all_training)

with open('date_process/test.pickle','wb') as fout:
	pickle.dump(all_training,fout)		

'''
PM 2.5 cleaning:

2014/3/12 [65.0, 61.0, 66.0, 66.0, 69.0, 131.0, 919.0, 919.0, 919.0, 0.0, 0.0, 908.0, 914.0, 914.0, 914.0, 70.0, 72.0, 82.0, 86.0, 97.0, 98.0, 98.0, 87.0, 76.0]
919.0 can't tell.
2014/4/2 [21.0, 22.0, 26.0, 15.0, 12.0, 3.0, 10.0, 13.0, 19.0, 16.0, 0.0, 631.0, 5.0, 0.0, 12.0, 15.0, 7.0, 10.0, 18.0, 29.0, 34.0, 37.0, 50.0, 51.0]
631.0 can't tell.

2014/4/6 NMHC / THC chaos.
'''