import keras
from keras import initializers
from keras.layers import *
from keras.models import *
from danager import danager
import sys
from keras.models import *
from keras.callbacks import *
user_idx = Input(shape=[1,])
movie_idx = Input(shape=[1,])
user_info = Input(shape=[30,])
movie_info = Input(shape=[18,])

user_output = Add()([Embedding(6041,300, input_length=[1])(user_idx) ,
	Embedding(6041,1, input_length=[1])(user_idx)])

movie_output = Add()([Embedding(3953,300, input_length=[1])(movie_idx),
	Embedding(3953,1, input_length=[1])(movie_idx)])


user_output = Flatten()(user_output)
movie_output = Flatten()(movie_output)

user_output = Dropout(0.5)(user_output)
movie_output= Dropout(0.5)(movie_output)
user_info_output = Dropout(0.5)(Dense(300)(user_info))
movie_info_output = Dropout(0.5)(Dense(300)(movie_info))

user = concatenate([user_output,user_info_output])
movie = concatenate([movie_output,movie_info_output])


user_output = Dense(1024,activation='tanh')(user)
movie_output= Dense(1024,activation='tanh')(movie)
user_output = Dropout(0.5)(user_output)
movie_output = Dropout(0.5)(movie_output)

y = Dot(1)([user_output , movie_output])
y = Dense(1)(y)
model = Model([user_idx , movie_idx , user_info , movie_info] ,y)

model.load_weights('model.hdf5')
danager = danager(mode='test' ,testfile=sys.argv[1])

predict = model.predict_generator(danager.test_generator(256) , steps=len(danager.test)//256+1 , verbose=0)

out = open(sys.argv[2],'w')
print('TestDataID,Rating' , file=out)

for idx,ans in enumerate(predict):
	if ans[0] < 1: 
		ans[0] = 1
	if ans[0] > 5:
		ans[0] = 5
	print('{},{}'.format(idx+1,ans[0]) , file=out)
'''
model.fit_generator(
	danager.train_generator(32),
	samples_per_epoch=len(danager.train) // 32,
	epochs=10,
	validation_data = danager.val_generator(32),
	validation_steps = len(danager.val) // 32, 
	verbose=1,
	callbacks=[checkpoint]
)
'''
