import gensim
import pickle
import numpy as np
import keras
import sys
from keras import initializers
from keras.layers import Input
from keras.models import load_model

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , LeakyReLU
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization , SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K



# emb_model = gensim.models.Word2Vec.load('little_gensim_embedding.model')
word_dict = pickle.load(open('word_dict','rb'))
test_emb = []
max_len = 40
print()
with open(sys.argv[1],'r') as fin:
	first = True
	for row in fin:
		if first:
			first=False
			continue
		row = ','.join(row.split(',')[1:]).split()
		t_emb = []
		for word in row:
			t_emb.append(word_dict[word])
		t_emb= np.array(t_emb)
		t_emb = np.concatenate([t_emb , np.zeros( (max_len-len(row),128))] , 0)
		test_emb.append(t_emb)
K = 4
split_size = int(len(test_emb)/K)
test_pool = []
for i in range(K):
	print(i*split_size ,(i+1)*split_size)
	test_pool.append(test_emb[i*split_size:(i+1)*split_size]) 

model = load_model('lstm_model.hdf5')
output = []
for this_test in test_pool:
	this_test = np.array(this_test).reshape(-1,40,128,1)
	output += model.predict(this_test , verbose=1).reshape(-1).tolist()
	# print(output)
fout = open(sys.argv[2],'w')

fout.write("id,label\n")
_ = 0
for row in output:
	print(row)
	fout.write('%d,%d\n'%(_,int(row>0.5)))
	_+=1
