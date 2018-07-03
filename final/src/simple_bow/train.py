import keras
import random
import gensim
import pickle
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
from danager import danager
from model import model
print('tf version:' , tf.__version__)
print('keras version:' , keras.__version__)
print('gensim version:' , gensim.__version__)
print('load danager..')
danager = danager()
random.seed(7122)
np.random.seed(7122)
model = model.simple_sim3()
print(model.summary())
optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

checkpoint = ModelCheckpoint('simple.hdf5', monitor='val_acc', verbose=1, save_best_only=True)

model.load_weights('simple.hdf5')
model.fit_generator(
	danager.datagen(64,mode='train'),
	samples_per_epoch= danager.train_len // 64,
	epochs=100,
	validation_data=danager.datagen(64,mode='val'),
	validation_steps= danager.val_len // 64 ,
	verbose=1,
	callbacks=[checkpoint]
)