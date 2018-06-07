import gensim
import random
import collections
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split

class dual_model:
	def __init__(self, model1 , model2):
		self.emb_model1 = gensim.models.Word2Vec.load(model1)
		self.emb_model2 = gensim.models.Word2Vec.load(model2)
	def __getitem__(self, key):
		out1 = self.emb_model1[key]
		out2 = self.emb_model2[key]
		out3 = []
		for a,b in zip(out1,out2):
			out3.append( np.concatenate([a,b],0) )
		return out3

class danager:
	def __init__(self , fixed_random_state = True):
		self.emb_model = dual_model('clean_data/cbow_gensim_embedding.model' , 'clean_data/skip_gram_gensim_embedding.model')
		self.train_raw_sentences = []
		self.semi_raw_sentences = []
		self.trainY = []
		with open('clean_data/training_label.txt','r') as fin:
			for row in fin:
				self.trainY.append(int(row.split('+++$+++')[0]))
				self.train_raw_sentences.append(row.split('+++$+++')[1].split())

		validation_split = 0.1
		if fixed_random_state:
			self.train_raw_sentences, self.train_val_sentences, self.trainY, self.valY = \
				train_test_split(self.train_raw_sentences, self.trainY, test_size=validation_split, random_state=7122)
		else:
			self.train_raw_sentences, self.train_val_sentences, self.trainY, self.valY = \
				train_test_split(self.train_raw_sentences, self.trainY, test_size=validation_split)
		
		self.valX = []
		for row in self.train_val_sentences:
			t_emb = self.emb_model[row]
			t_emb = np.concatenate([t_emb , np.zeros( (40-len(row),256))] , 0)
			self.valX.append(t_emb)
		self.valX = np.array(self.valX).reshape(-1,40,256,1)
		self.valY = np.array(self.valY).reshape(-1,1)
		with open('clean_data/training_nolabel.txt','r') as fin:
			for row in fin:
				if row.split() and len(row.split()) <= 40:
					self.semi_raw_sentences.append(row.split())
		self.length = len(self.train_raw_sentences)
		self.semi_length = len(self.semi_raw_sentences)
	def data_generator(self,batch_size):
		while True:
			idx = np.random.randint(0,len(self.train_raw_sentences),size=[batch_size])
			X = []
			Y = []
			for now_idx in idx:
				row = self.train_raw_sentences[now_idx]
				t_emb = self.emb_model[row]
				t_emb = np.concatenate([t_emb , np.zeros( (40-len(row),256))] , 0)
				X.append(t_emb)
				Y.append(self.trainY[now_idx])
			X = np.array(X).reshape(-1,40,256,1)
			Y = np.array(Y).reshape(-1,1)
			yield X,Y
	def validation_set(self):
		return [self.valX , self.valY]

	def semi_data_generator(self,batch_size):
		
		self.semi_counter = 0
		while True:
			X , Y = [] , []
			for _ in range(batch_size):
				if self.semi_counter >= len(self.semi_raw_sentences):
					self.semi_counter = 0
				row = self.semi_raw_sentences[self.semi_counter]
				t_emb = self.emb_model[row]
				t_emb = np.concatenate([t_emb , np.zeros( (40-len(row),256))] , 0)
				X.append(t_emb)
				self.semi_counter += 1
			X = np.array(X).reshape(-1,40,256,1)
			yield X
	def semi_transfer(self,predict,transfer_size):
		predict = np.array(predict).reshape(-1)
		extremety = (1-predict) * (predict)
		rank= np.sort(np.argsort(extremety)[:transfer_size])
		predict = (predict[rank] > 0.5).astype(np.int)
		deq =  deque(rank)
		temp = []
		new_member = []
		for idx,sen in zip(range(self.semi_length) , self.semi_raw_sentences):
			if deq and idx == deq[0]:
				deq.popleft()
				new_member.append(sen)
			else:
				temp.append(sen)
		self.semi_raw_sentences = temp
		self.train_raw_sentences += new_member
		self.trainY += predict.tolist()
		self.length += transfer_size
		self.semi_length -= transfer_size
		self.semi_counter = 0

	def test_data_generator(self,batch_size):		
		test_sen = []
		max_len = 40
		with open('clean_data/testing_data.txt','r') as fin:
			first = True
			for row in fin:
				if first:
					first=False
					continue
				row = ','.join(row.split(',')[1:]).split()
				test_sen.append(row)

		X = []
		for counter , row in zip(range(len(test_sen)),test_sen):
			t_emb = self.emb_model[row]
			t_emb = np.concatenate([t_emb , np.zeros( (40-len(row),256))] , 0)
			X.append(t_emb)
			if counter%batch_size == batch_size-1:
				X = np.array(X).reshape(-1,40,256,1)
				yield X
				X = []