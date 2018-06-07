import keras
import random
import gensim
import pickle
import numpy as np
from keras import backend as K
from keras import initializers
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
from no_label_process import no_label_generator
from bidanager import danager
from lstm_model import model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
'''

print('tf version:' , tf.__version__)
print('keras version:' , keras.__version__)
print('gensim version:' , gensim.__version__)


danager = danager()
random.seed(7122)
np.random.seed(7122)
semi_training  = False

model = model()
print(model.summary())
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpoint = ModelCheckpoint('two_emb_bilstm_rcnn_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True)

#model = load_model('large_cnn2.hdf5')

for i in range(40):
	print('now training_length:{} , semi_length:{}'.format(danager.length,danager.semi_length))
	model.fit_generator(
		danager.data_generator(32),
		samples_per_epoch=danager.length // 32,
		epochs=2*(i+1),
		validation_data=danager.validation_set(),
		verbose=1,
		initial_epoch=2*i,
		callbacks=[checkpoint]
	)
	if semi_training:
		output = model.predict_generator(
			danager.semi_data_generator(512),
			steps = danager.semi_length // 512,
			workers = 8 ,
			verbose = 1
		)
		if danager.semi_length >= 100000:
			danager.semi_transfer(output,100000)
