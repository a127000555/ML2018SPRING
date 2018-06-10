import pickle
import numpy as np
class danager:
	def __init__(self , mode='train' , testfile='test.csv'):
		self.movie_feature=pickle.load(open('data/movies.pickle','rb'))
		self.user_feature=pickle.load(open('data/users.pickle','rb'))
		# self.critique_feature=pickle.load(open('data/critique.pickle','rb'))
		if mode=='train':
			train=pickle.load(open('data/train.pickle','rb'))[:,1:]
			idx = list(range(len(train)))
			np.random.shuffle(idx)
			validation_split = 0.05
			self.train , self.val = train[idx[int(len(train)*validation_split):]],train[idx[:int(len(train)*validation_split)]]
		elif mode=='test':
			self.test = [row.split(',') for row in open(testfile,'r')][1:]
			self.test = np.array(self.test).astype(np.int32)
		else:
			print('what do u want?')
	def get_info(self,userID,movieID):
		
		if movieID not in self.movie_feature:
			print("{} movie not found. I'll give you all zero matrix".format(movieID))
			mv_feat = np.zeros([18])
		else:
			mv_feat = self.movie_feature[movieID]
		if userID not in self.user_feature:
			print("{} user not found. I'll give you all zero matrix".format(userID))
			us_feat = np.zeros([18])
		else:
			us_feat = self.user_feature[userID]
		return [userID] , [movieID] , us_feat , mv_feat

	def data_generator(self , batch_size , mode='train'):
		while True:
			if mode == 'train':
				chosen_data = self.train
			elif mode== 'val':
				chosen_data = self.val
			else:
				print('unknown mode:' , mode)
				exit()

			idx = np.random.randint(len(chosen_data),size=[batch_size])
			trainY= []
			feature = [ [] for i in range(4)]
			for now_idx in idx:
				userID , MovieID , rating = chosen_data[now_idx]
				this_feat = self.get_info(userID,MovieID)
				for idx , d in enumerate(this_feat):
					feature[idx].append(d)
				trainY.append([rating])
			for i in range(4):
				feature[i] = np.array(feature[i])
			trainY = np.array(trainY)
			yield feature, trainY
	def test_generator(self,batch_size):
		counter = -batch_size

		# for counter in range(0,len(self.test) , batch_size):
		while True:
			counter += batch_size
			idx = range(counter , min(counter+batch_size,len(self.test)))
			print(idx)
			feature = [ [] for i in range(4)]
			for now_idx in idx:
				testID,userID , MovieID = self.test[now_idx]
				this_feat = self.get_info(userID,MovieID)
				for idx , d in enumerate(this_feat):
					feature[idx].append(d)
			for i in range(4):
				feature[i] = np.array(feature[i])
			yield feature
		
if __name__ == '__main__':
	D = danager(mode='test')
	i = 0
	for row in D.test_generator(32):
		pass