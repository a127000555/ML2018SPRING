import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
train = pickle.load(open('date_process/train.pickle','rb'))


count = []
for day in train:
	row = day['RAINFALL']
	count += row
from collections import Counter
counter = Counter(count)
for  c in counter.most_common(1000):
	print(c)



plt.legend(loc='upper right')

#plt.show()