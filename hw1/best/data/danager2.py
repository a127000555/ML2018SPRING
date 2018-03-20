import csv
import numpy as np 
import pickle

def train_preprocess():
	cin = csv.reader(open('train.csv','r',encoding='big5'))
	table = [row for row in cin]

	concatenate_table ,now_data = [],[ [table[1+_][2]] for _ in range(18)]
	last_date = 0
	for row_idx_head in range(1,len(table),18):
		now_date = int(table[row_idx_head][0].split('/')[2]) 

		if last_date+1 != now_date:
			concatenate_table.append(now_data)
			now_data = [ [table[row_idx_head+_][2]] for _ in range(18)]
			last_date = 0

		for measure_idx in range(0,18):
			row = table[row_idx_head + measure_idx][3:]
			#print(row[0] , row[1] , row[2])
			row =[ '0.0' if x == 'NR' else x for x in row ]
			now_data[measure_idx] += row
		last_date += 1
	else:
		concatenate_table.append(now_data)
	
	#for timeline in concatenate_table:
	#	print(timeline[1][0])
	#print(concatenate_table)
	with open('data_process(time_cat)/train.pickle','wb') as fout:
		pickle.dump(concatenate_table,fout)		

def test_preprocess():
	cin = csv.reader(open('test.csv','r',encoding='big5'))
	table = [row for row in cin]
	
	all_training = []
	for row_idx_head in range(0,len(table),18):
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
		

		# PM2.5 / PM10 error correct
		for fa in ['PM2.5' , 'PM10']:
			pm = list(map(float,day[fa]))
			for idx in range(len(pm)):
				if float(pm[idx]) < 2  or float(pm[idx]) > 120:
					if idx != 0 and idx != len(pm)-1:
						print(idx)
						pm[idx] = (pm[idx-1]+pm[idx+1])/2
					if idx == 0:
						pm[idx] = (pm[idx+1] + pm[idx+2]) / 2
					if idx == len(pm)-1:
						pm[idx] = (pm[idx-1] + pm[idx-2]) /2

					#= np.float('Nan')
			
			pm = np.array(pm)
			if np.isnan(pm).any():
				print(pm[pm != np.nan])
				print(np.mean(pm))

				exit(0)
			day.update( {fa : pm})

		all_training.append(day)

	with open('data_process(time_cat)/test.pickle','wb') as fout:
		pickle.dump(all_training,fout)			


def generate():
	train_in = pickle.load(open('data_process(time_cat)/train.pickle','rb'))
	
	train_data = []
	train_label = []
	test_data = []

	### param ###
	time_range = 9
	offset = 0
	concentrate = ['PM10' , 'PM2.5' ]
	#############
	
	## training prepare
	for timeline in train_in:
		con_idx = []
		for _ in range(len(timeline)):
			if timeline[_][0] in concentrate:
				con_idx.append(_)
		for left in range(1,len(timeline[0])-time_range-1):
			section = []
			# CO / PM10 / PM2.5
			try:
				for idx in con_idx:
					factor = timeline[idx][0]
					row = timeline[idx][1:]
					np_row = np.array(row[left:left+time_range]).astype('float32')
					if factor == 'PM2.5' or factor == 'PM10':
						if len(np_row[np_row<2]):
							raise Error()
						if len(np_row[np_row>120]):
							raise Error()
						#print(np_row)
					elif factor == 'CO':
						pass
						#print(np_row)
					section += row[left+offset:left+time_range] 
				#print()
			except:
				continue
			train_data.append(np.array(section).astype(np.float))
			train_label.append(np.array([row[left+time_range]]).astype(np.float))
	
	## testing for training prepare
	test_in = pickle.load(open('data_process(time_cat)/test.pickle','rb'))

	for day in test_in :
		con_arr = []
		for i in concentrate:
			con_arr.append(np.array(day[i]).astype(np.float))

		N = len(con_arr[0])
		for i in range(N-time_range+1):
			# CO / PM10 / PM2.5
			data = []
			for fa in con_arr:
				#data.append( np.mean(np.array(fa[i:i+time_range]).astype(np.float)))	
				data += list(fa[i+offset:i+time_range]) 
				#print(i , fa)
			if i == N-time_range:
				test_data.append(data)
			else:
				y = [con_arr[0][i+time_range]]
				if np.isnan(data).any() or np.isnan(y).any():
					print(data)
					continue

				train_data.append(data)
				train_label.append(y)

			
	print(np.array(train_data).shape)
	print(np.array(train_label).shape)
	print(np.array(test_data).shape)
	for r in train_data:
		print(r)
	dir_name = 'pm25_pm10(time_cat)'
	np.save(dir_name + '/' + 'train_data' , (np.array(train_data)))
	np.save(dir_name + '/' + 'train_label' ,  (np.array(train_label)))
	np.save(dir_name + '/' + 'test_data' ,  (np.array(test_data)))
	
	#for _ in range(10):
	#	print(train_data[_])
	

if __name__ == '__main__' :
	train_preprocess()
	test_preprocess()
	generate()
