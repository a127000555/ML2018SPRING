from gensim.models import word2vec
import gensim
import jieba
import numpy as np
import random
random.seed(12123)
np.random.seed(12123)
emb_model =  gensim.models.Word2Vec.load('word_emb')
	
def sp(sen):
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

corpus = {}
train = []

now_idx = 0
for i in range(1,6):
	fin = open('training_data/' + str(i) + '_train.txt','r')
	begin = True
	for row in fin:
		if len(row) == 0:
			begin = True
			continue
		corpus[now_idx] = sp(row)
		if begin == False:
			train.append([now_idx-1 ,now_idx , 1])
		
		now_idx += 1
		begin = False
	print('read train' + str(i))
'''
N = len(train)
new_train = []
for row in train:	
	rand_choice = [ random.randint(0,len(corpus)-1) for i in range(5	)] + [row[1]]
	random.shuffle(rand_choice)
	row = [ [row[0]] ,  rand_choice , [rand_choice.index(row[1])] ]
	new_train.append(row)
# print(train)
train = new_train
'''
random.shuffle(train)

import  pickle
pickle.dump(train,open('train','wb') )
pickle.dump(corpus,open('corpus','wb') )