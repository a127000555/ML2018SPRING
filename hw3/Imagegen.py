import numpy as np
from keras.preprocessing.image import ImageDataGenerator
def datagen(image , label, batch_size = 32):
	
	datagen = ImageDataGenerator(
	    rotation_range=10,
	    width_shift_range=0.3,
	    height_shift_range=0.3,
	    horizontal_flip = True,
	    channel_shift_range=0.3,
	    zoom_range=0.2
	)
	datagen.fit(image)
	return datagen.flow(image, label, batch_size=batch_size)

def severe_datagen(image , label, batch_size = 32):
	datagen = ImageDataGenerator(
	    rotation_range=30,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    zoom_range=[0.8, 1.2],
	    shear_range=0.2,
		horizontal_flip=True)
	datagen.fit(image)
	return datagen.flow(image, label, batch_size=batch_size)


def my_shuffle(X,Y):
	idx = list(range(len(X)))
	import random
	random.shuffle(idx)
	return X[idx] , Y[idx]