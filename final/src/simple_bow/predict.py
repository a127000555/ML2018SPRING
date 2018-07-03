import pandas as pd
import numpy as np
import jieba
import sklearn
import gensim
from gensim.models import word2vec
from keras.models import *
from scipy.spatial.distance import cosine
df = pd.read_csv('testing_data.csv')
data = np.array(df)
def sp(sen):
	global word_pool
	next_sen = []
	for txt in sen:
		if txt in ['，' , '、','\t' , '.' , '?']:
			txt = ' '
		next_sen.append(txt)
	sen = ''.join(next_sen)
	next_sen = []
	for word in sen.split():
		seg_list = jieba.cut(word, cut_all=False)
		for single_word in seg_list:
			next_sen.append(single_word)
	return next_sen

emb_model =  gensim.models.Word2Vec.load('word_emb')
A,B,C = [],[],[]

def zero_fill(A):
	A = np.array(A)
	pad = np.zeros([40 - len(A) , 64])
	return np.concatenate([A,pad],0)
m = 0
for data_idx , row in zip(range(6000),data):
	target = zero_fill(emb_model[sp(row[1])])
	t = []
	for i , opt in zip(range(6),row[2].split('\t')):
		choice = zero_fill(emb_model[sp(opt[2:])])
		A.append(target)
		B.append(choice)
		C.append([1])
A = np.array(A)
B = np.array(B)
C = np.array(C)
print(A.shape,B.shape,C.shape)

from model import model
model = model.simple_sim3()
model.load_weights('simple.hdf5')
output = model.predict([A,B,C])
fout = open('sim3.csv', 'w')
print('id,ans' , file=fout)

data_idx = 0
for i in range(0,len(output),6):
	print(output[i:i+6])
	idx = np.argmax(output[i:i+6])
	print('{},{}'.format(data_idx,idx) , file=fout)
	data_idx += 1