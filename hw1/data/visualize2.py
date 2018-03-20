import pandas as pd
import numpy as np
file_list = ['train.csv']
df = pd.read_csv(file_list[0],encoding='big5')
print(type(df))


import csv 
cin = csv.reader(open(file_list[0],'r',encoding='big5'))
raw = list(filter(lambda t : 'PM2.5' in t ,[row for row in cin] ) )

for row in raw:
	for item in row[3:]:
		print( "%4s" % (item) , end=' ')
	print()