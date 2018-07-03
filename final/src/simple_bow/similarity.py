import pandas as pd
import numpy as np
import jieba
import sklearn
import gensim
from gensim.models import word2vec
from scipy.spatial.distance import cosine
df = pd.read_csv('testing_data.csv')
data = np.array(df)
sentences = []
word_pool = set()
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
			word_pool.add(single_word)
			next_sen.append(single_word)
	return next_sen

def add(sen):
	next_sen = sp(sen)
	sentences.append(next_sen)
for row in data:
	add(row[1])
	for opt in row[2].split('\t'):
		add(opt[2:])

print('read test data')
T = 10
for i in range(1,6):
	fin = open('training_data/' + str(i) + '_train.txt','r')
	last = [ '' for i in range(10)]
	for row in fin:
		add(' '.join(last) + ' ' + row)
		last  = [ last[i+1] for i in range(9)] + [row]
	print('read train' + str(i))
print(len(word_pool))
print('start training')
model = gensim.models.Word2Vec(sentences, min_count=1 , size=64 , workers = 8 , iter=100)
model.save('word_emb')
print('end training')
'''
fout = open('ans.csv', 'w')
print('id,ans' , file=fout)

for data_idx , row in zip(range(6000),data):
	target = np.sum(model[sp(row[1])],0)
	idx , now = 0 , 100
	for i , opt in zip(range(6),row[2].split('\t')):
		dis = cosine(target,np.sum(model[sp(opt[2:])],0))
		if now > dis:
			idx , now = i , dis
	print('{},{}'.format(data_idx,idx) , file=fout)
'''
