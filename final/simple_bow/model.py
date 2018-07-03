import gensim
import pickle
import numpy as np
import keras
from keras import initializers
from keras.models import *
from keras.layers import *
from keras import backend as K
import tensorflow as tf
class model:
	def simple_lstm():
		A = Input(shape=(13,64))
		B = Input(shape=(5,13,64))
		weight = Input(shape=(1,))
		LSTM1 = LSTM(64 , dropout=0.0, recurrent_dropout=0.3, return_sequences=True)
		BN1 = BatchNormalization()
		LSTM2 = LSTM(128, dropout=0.0, recurrent_dropout=0.3, return_sequences=False)
		BN2 = BatchNormalization()
		DS1= Dense(756)
		BN3 = BatchNormalization()
		DS2= Dense(256)
		BN4 = BatchNormalization()
		
		def LSTM_out(output):
			output = LSTM1(output)
			output = LeakyReLU(0.1)(output)
			output = BN1(output)
			
			output = LSTM2(output)
			output = LeakyReLU(0.1)(output)
			output = BN2(output)

			output = LeakyReLU(0.1)(output)
			output = Dropout(0.3)(output)
			output = DS1(output)
			output = BN3(output)

			output = LeakyReLU(0.1)(output)
			output = Dropout(0.3)(output)
			output = DS2(output)
			output = BN4(output)

			return output
		A_out = LSTM_out(A)
		output = []
		for i in range(5):
			x = Lambda(lambda x : x[:,i,:,:])(B)
			B_out = LSTM_out(x)
			output.append(Dot(1, normalize=True)([A_out,B_out]))
		
		output = Concatenate(axis=-1)(output)
		y = Dense(5, activation='softmax')(output)
		model = Model([A,B,weight],y)
		return model
	def simple_sim():
		A = Input(shape=(40,64))
		B = Input(shape=(6,40,64))
		weight = Input(shape=(1,))
		
		A_out = Dense(2048 , kernel_initializer=initializers.lecun_uniform())(A)
		A_out = Dropout(0.3)(A_out)
		A_out = LeakyReLU(0.1)(A_out)
		A_out = BatchNormalization()(A_out)

		B_out = Dense(2048, kernel_initializer=initializers.lecun_uniform())(B)

		A_out = Lambda(lambda x : tf.reduce_sum(x,1))(A_out)
		B_out = Lambda(lambda x : tf.reduce_sum(x,2))(B_out)

		output = []
		for i in range(6):
			x = Lambda(lambda x : x[:,i,:])(B_out)

			x = Dropout(0.3)(x)
			x = LeakyReLU(0.1)(x)
			x = BatchNormalization()(x)

			output.append(Dot(1, normalize=True)([A_out,x]))
		
		output = Concatenate(axis=-1)(output)
		
		y = Dense(6, activation='softmax')(output)
		model = Model([A,B,weight],y)
		return model
	def simple_sim2():
		A = Input(shape=(40,64))
		B = Input(shape=(40,64))
		weight = Input(shape=(1,))
		
		A_out = Dense(256 , kernel_initializer=initializers.lecun_uniform())(A)
		A_out = Dropout(0.3)(A_out)
		A_out = LeakyReLU(0.1)(A_out)
		A_out = BatchNormalization()(A_out)
		A_out = Dense(2048 , kernel_initializer=initializers.lecun_uniform())(A_out)
		A_out = Dropout(0.3)(A_out)
		A_out = LeakyReLU(0.1)(A_out)
		A_out = BatchNormalization()(A_out)

		B_out = Dense(256, kernel_initializer=initializers.lecun_uniform())(B)
		B_out = Dropout(0.3)(B_out)
		B_out = LeakyReLU(0.1)(B_out)
		B_out = BatchNormalization()(B_out)
		B_out = Dense(2048, kernel_initializer=initializers.lecun_uniform())(B_out)
		B_out = Dropout(0.3)(B_out)
		B_out = LeakyReLU(0.1)(B_out)
		B_out = BatchNormalization()(B_out)

		A_out = Lambda(lambda x : tf.reduce_sum(x,1))(A_out)
		B_out = Lambda(lambda x : tf.reduce_sum(x,1))(B_out)

		y = Dot(1, normalize=True)([A_out,B_out])
		
		y = Dense(1, activation='sigmoid')(y)
		model = Model([A,B,weight],y)
		return model

	def simple_sim3():
		A = Input(shape=(40,64))
		B = Input(shape=(40,64))
		weight = Input(shape=(1,))
		
		A_out = Dense(1024 , kernel_initializer=initializers.lecun_uniform())(A)
		A_out = Dropout(0.2)(A_out)
		A_out = LeakyReLU(0.1)(A_out)
		A_out = BatchNormalization()(A_out)
		A_out = Dense(4096 , kernel_initializer=initializers.lecun_uniform())(A_out)
		A_out = Dropout(0.3)(A_out)
		A_out = LeakyReLU(0.1)(A_out)
		A_out = BatchNormalization()(A_out)

		B_out = Dense(1024, kernel_initializer=initializers.lecun_uniform())(B)
		B_out = Dropout(0.2)(B_out)
		B_out = LeakyReLU(0.1)(B_out)
		B_out = BatchNormalization()(B_out)
		B_out = Dense(4096, kernel_initializer=initializers.lecun_uniform())(B_out)
		B_out = Dropout(0.3)(B_out)
		B_out = LeakyReLU(0.1)(B_out)
		B_out = BatchNormalization()(B_out)

		A_out = Lambda(lambda x : tf.reduce_sum(x,1))(A_out)
		B_out = Lambda(lambda x : tf.reduce_sum(x,1))(B_out)

		y = Dot(1, normalize=True)([A_out,B_out])
		y = Dense(1, activation='sigmoid')(y)
		model = Model([A,B,weight],y)
		return model
		
if __name__ == '__main__':
	print(model.simple_sim2().summary())