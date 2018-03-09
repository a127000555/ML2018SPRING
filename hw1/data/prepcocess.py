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
		if rain[idx] == 'NR':
			rain[idx]  = '0.0'
	day.update( {'RAINFALL' : rain})
	
	# NMHC cleaning:
	nmhc = day['NMHC']
	for idx in range(len(nmhc)):
		if  float(nmhc[idx]) > 4:
			print(day['id'])
			exit(0)
			nmhc[idx] = '2.5'
	day.update( {'NMHC' : nmhc})
	
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
	# RAINFALL concert
	rain = day['RAINFALL']
	for idx in range(len(rain)):
		if rain[idx] == 'NR':
			rain[idx]  = '0.0'
	day.update( {'RAINFALL' : rain})
	
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