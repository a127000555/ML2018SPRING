import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import math
train = pickle.load(open('date_process/train.pickle','rb'))

X = []
Y = []
for day in train:
	temp = day['id'].split('/')
	time_line = (int(temp[1]) * 31) + (int(temp[2]))
	for pm , temp in zip(np.array(day['PM2.5']).astype(np.float) ,np.array(day['AMB_TEMP']).astype(np.float) ):
		# 0: cold  40 : hot
		#print(time_line , pm)
		red = hex(int(temp / 50 * 256) )
		blue = hex(int( (50-temp) / 50 * 256) )
		clr = "#" + red[2:] + "00" + blue[2:]
		try:
			plt.scatter(x=time_line,y=pm,color=clr)
		except:
			print(clr , blue)

	X.append(time_line)
	Y.append(pm)

plt.show()