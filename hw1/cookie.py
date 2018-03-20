import numpy as np
import csv
import sys
testin = 'test.csv'
testout = 'ans.csv'
if len(sys.argv)==3:
	print("get arguments.")
	testin = sys.argv[1]
	testout = sys.argv[2]
w = '''-7.69261346e-04  -2.84478584e-02   4.18044901e-02  -3.27880533e-02
   -8.70547403e-02   1.23435240e-01  -6.19149040e-02  -1.25339963e-01
	2.25940524e-01   1.49998624e-02  -2.33600381e-02   2.24084668e-01
   -2.38700767e-01  -9.60996634e-03   5.23019265e-01  -5.77325181e-01
	2.09961766e-02   9.66783180e-01'''.split()
w = np.array(w).astype(np.float)

# test data process #
table = [row for row in csv.reader(open(testin,'r',encoding='big5'))]
table = list(filter(lambda s : 'PM2.5' in s or 'PM10' in s , table))
outtable = [['id','value']]
for row_idx in range(0,len(table),2):
	test_id = table[row_idx][0]
	x = []
	for factor in [ table[row_idx] ,table[row_idx+1] ]:
		pm = list(map(np.float,factor[2:]))
		for idx in range(len(pm)):
			if float(pm[idx]) < 2  or float(pm[idx]) > 120:
				if idx != 0 and idx != len(pm)-1:
					pm[idx] = (pm[idx-1]+pm[idx+1])/2
				if idx == 0:
					pm[idx] = (pm[idx+1] + pm[idx+2]) / 2
				if idx == len(pm)-1:
					pm[idx] = (pm[idx-1] + pm[idx-2]) /2
		x += pm
	outtable.append( [ test_id , np.dot(w,x)])

csv.writer(open(testout,'w')).writerows(outtable)