import numpy as np
from sklearn.model_selection import train_test_split
image = np.load('trainX.npy')
label = np.load('trainY.npy')

#for CNNy

image = image.reshape(-1,48,48,1)
spl = int(len(image)*0.9)
idx = list(range(len(image)))

import random 
random.shuffle(idx)
image= image[idx]
label= label[idx]


valX = image[spl:]
valY = label[spl:]
image= image[:spl]
label= label[:spl]

print(np.shape(image),np.shape(label))
import keras
from keras import initializers
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , LeakyReLU
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 5), input_shape=[48,48,1], padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#model.add(Dropout(0.3))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#model.add(Dropout(0.35))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
#model.add(BatchNormalization())
#odel.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('model_init.hdf5', monitor='val_acc', save_best_only=True, verbose=1)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip = True,
    channel_shift_range=0.3,
    zoom_range=0.2
)

from keras.models import load_model
#model = load_model('model_init.hdf5')
datagen.fit(image)

model.fit_generator(
	datagen.flow(image, label, batch_size=64),
	steps_per_epoch=len(image)/16, 
	epochs=10,
	validation_data=[valX,valY],
	shuffle=True,		
	verbose=1,
	callbacks=[checkpoint]
	)

model.save('model')

model = load_model('model_init.hdf5')
output = model.predict(np.load('testX.npy').reshape(-1,48,48,1))

fout = open('ans.csv','w')

fout.write("id,label\n")
_ = 0
for row in output:
	fout.write('%d,%d\n'%(_,np.argmax(row)))
	_+=1