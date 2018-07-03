from gensim.models import word2vec
import gensim
import jieba
import numpy as np
import random
import pickle
class danager:
	def __init__(self):
		self.emb_model =  gensim.models.Word2Vec.load('word_emb')
		self.corpus = pickle.load(open('corpus','rb'))
		self.data = pickle.load(open('train','rb'))
		self.max_length = 40
		self.word_emb = 64
		idx = np.arange(0,len(self.data))
		split_rate = 0.15
		np.random.shuffle(idx)
		self.train_idx = idx[int(len(self.data) * split_rate) :]
		self.val_idx = idx[: int(len(self.data) * split_rate)]

		self.train_len = len(self.train_idx)
		self.val_len = len(self.val_idx)
		print(self.train_len)
	def idx_to_emb(self,idx):
		raw_sen = self.corpus[idx]
		emb = self.emb_model[raw_sen]
		return self.zero_fill(emb)
	def zero_fill(self,A):
		A = np.array(A)
		pad = np.zeros([self.max_length - len(A) , self.word_emb])
		return np.concatenate([A,pad],0)

	def datagen(self , bt_size =32 , mode = 'train'):
		if mode == 'train':
			idx_pool = self.train_idx
		elif mode == 'val':
			idx_pool = self.val_idx
		else:
			print('what do you want?')
			exit()
		while True:
			sen1,sen2 , y , weight = [] ,[], [], []
			for _ in range(bt_size):
				j = random.randint(0,len(idx_pool) -1)
				A = self.idx_to_emb(self.data[idx_pool[j]][0])
				if np.random.uniform() < 0.5:
					B = self.idx_to_emb(self.data[idx_pool[j]][1])
					C = [1]
				else:
					B = self.idx_to_emb(random.randint(0,len(self.corpus) -1))
					C = [0]
				D = [ self.data[idx_pool[j]][2] ]
				sen1.append(A)
				sen2.append(B)
				y.append(C)
				weight.append(D)
			yield [np.array(sen1) , np.array(sen2), np.array(weight)] , np.array(y) 

if __name__ == '__main__':
	np.random.seed(12123)
	D = danager()
	for x in D.datagen():
		print(x[0][0].shape,x[0][1].shape,x[0][2].shape,x[1].shape,)