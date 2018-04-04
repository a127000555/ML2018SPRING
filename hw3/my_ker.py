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


### DATA ###
image = np.load('trainX.npy').reshape(-1,48,48,1)
label = np.load('trainY.npy')
validation_split = 0.1
train_X, test_X, train_Y, test_Y = train_test_split(image, label, test_size=validation_split, random_state=127)

### MACRO ###
def conv(filters , size , strides = (1,1)):
	return Conv2D(filters, strides = strides , kernel_size=(size, size),padding='same', kernel_initializer='he_normal')

def pool():
	return MaxPooling2D(pool_size=(2, 2), padding='same')

def den(neurons):
	return Dense(neurons, activation='relu', kernel_initializer='glorot_normal')

### MODEL ###
x = Input(shape=(48,48,1))

conv1_1 = Dropout(0.3)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(32,5)(x))))
conv1_2 = Dropout(0.3)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(32,5)(conv1_1))))
conv1_3 = Dropout(0.3)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(32,5)(conv1_2))))
pool1 = Dropout(0.3)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(32,5,strides=[2,2])(keras.layers.Add()([conv1_1 , conv1_3])))))

conv2_1 = Dropout(0.35)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(64,3)(pool1))))
conv2_2 = Dropout(0.35)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(64,3)(conv2_1))))
conv2_3 = Dropout(0.35)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(64,3)(conv2_2))))
pool2 = Dropout(0.35)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(64,3,strides=[2,2])(keras.layers.Add()([conv2_1 , conv2_3])))))

conv3_1 = Dropout(0.4)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(128,3)(pool2))))
conv3_2 = Dropout(0.4)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(128,3)(conv3_1))))
conv3_3 = Dropout(0.4)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(128,3)(conv3_2))))
pool3 = Dropout(0.4)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(128,3,strides=[2,2])(keras.layers.Add()([conv3_1 , conv3_3])))))

conv4_1 = Dropout(0.45)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(256,3)(pool3))))
conv4_2 = Dropout(0.45)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(256,3)(conv4_1))))
conv4_3 = Dropout(0.45)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(256,3)(conv4_2))))
pool4 = Dropout(0.45)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(256,3,strides=[2,2])(keras.layers.Add()([conv4_1 , conv4_3])))))

#avg_pool = keras.layers.AveragePooling2D(3,3)(pool4)
#flat = Flatten()(avg_pool)
flat = Flatten()(pool4)


dense = Dropout(0.5)(BatchNormalization()(den(1024)(flat)))
y = Dense(7, activation='softmax', kernel_initializer='glorot_normal')(dense)


model = Model(x,y)
print(model.summary())
### COMPILE ###
model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('my_model.hdf5', monitor='val_acc', save_best_only=True, verbose=1)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip = True,
    channel_shift_range=0.3,
    zoom_range=0.2
)

from keras.models import load_model
datagen.fit(train_X)
#model = load_model('my_model.hdf5')
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

model = load_model('my_model.hdf5')
output = model.predict(np.load('testX.npy').reshape(-1,48,48,1))

fout = open('ans.csv','w')

fout.write("id,label\n")
_ = 0
for row in output:
	fout.write('%d,%d\n'%(_,np.argmax(row)))
	_+=1
