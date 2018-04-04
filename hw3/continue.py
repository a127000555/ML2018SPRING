from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras import initializers
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , LeakyReLU
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
### DATA ###
image = np.load('trainX.npy').reshape(-1,48,48,1)
label = np.load('trainY.npy')
validation_split = 0.1
train_X, test_X, train_Y, test_Y = train_test_split(image, label, test_size=validation_split, random_state=721)#127)

checkpoint = ModelCheckpoint('my_model.hdf5', monitor='val_acc', save_best_only=True, verbose=1)
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip = True,
    channel_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(train_X)
model = load_model('68041.hdf5')
print(model.summary())
model.fit_generator(
	datagen.flow(train_X, train_Y, batch_size=64),
	steps_per_epoch=len(train_X)/16, 
	epochs=200,
	validation_data=[test_X ,test_Y],
	shuffle=True,		
	verbose=1,
	callbacks=[checkpoint]
	)
model.save('con_68041.hdf5')
