#from sklearn.model_selection import train_test_split
import numpy as np
import keras
import sys
import csv
from keras import initializers
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , LeakyReLU
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization , SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

### DATA ###

trainfile = 'train.csv'
if len(sys.argv) > 1:
	trainfile = sys.argv[1]
 
train = [ row for row in csv.reader(open(trainfile,'r'))][1:]
image = []
label = []
test_image = []
for row in train:
	l = np.zeros((7))
	l[int(row[0])] = 1
	image.append(row[1].split())
	label.append(l)
image = np.array(image).astype(np.float).reshape(-1,48,48,1)
label = np.array(label).astype(np.float)

image = (image - np.mean(image,0)) / np.std(image,0)
validation_split = 0.2
#train_X, test_X, train_Y, test_Y = train_test_split(image, label, test_size=validation_split, random_state=7122)

train_X = image
test_X = image
train_Y = label
test_Y = label

### MACRO ###
def activation():
	return LeakyReLU()
	#return keras.layers.PReLU()
	#return Activation('relu')

def simple_cnn_v2(inputs , filters , filters_size , strides = (1,1) , padding='same' , g = False):
	if g == False:
		return 	activation()(
				BatchNormalization(axis=-1, momentum=0.5)(
					Conv2D(filters, 
						strides = strides ,
						kernel_size=filters_size , 
						padding=padding)(inputs)))
	else:
		return 	activation()(
					BatchNormalization(axis=-1, momentum=0.5)(
						keras.layers.GaussianNoise(0.1)(
						Conv2D(filters, 
							strides = strides ,
							kernel_size=filters_size , 
							padding=padding)(inputs))))
			
### MODEL ###
x = Input(shape=(48,48,1))
output = x 
######################################

output  = simple_cnn_v2(output,16,(3,3))
output  = simple_cnn_v2(output,32,(3,3),g=True)
output  = simple_cnn_v2(output,64,(3,3))
output  = MaxPooling2D((2,2),padding='same')(output)
outout  = Dropout(0.1)(output)
output  = simple_cnn_v2(output,128,(3,3))
output  = MaxPooling2D((2,2),padding='same')(output)
outout  = Dropout(0.2)(output)
output  = simple_cnn_v2(output,256,(3,3))
output  = MaxPooling2D((2,2),padding='same')(output)
outout  = Dropout(0.2)(output)
output  = simple_cnn_v2(output,512,(3,3),padding='same')
output  = MaxPooling2D((2,2))(output)
outout  = Dropout(0.2)(output)
######################################
output = Flatten()(output)
#output = Dense(512 , activation='maxout')(output)
output = Dense(512 )(output)
output = BatchNormalization()(output)
output = activation()(output)
outout  = Dropout(0.5)(output)
output = Dense(256)(output)
output = activation()(output)
#output = keras.layers.GlobalMaxPooling2D()(output)
y = Dense(7, activation='softmax')(output)


model = Model(x,y)

### COMPILE ###
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#print(model.summary())
#exit()
from keras.callbacks import ModelCheckpoint
#checkpoint = ModelCheckpoint('Inception_v1_model.02-0.46018.hdf5', monitor='val_acc', save_best_only=True, verbose=1)
checkpoint = ModelCheckpoint('report4.hdf5', monitor='val_acc', verbose=1, save_best_only=True)


datagen = ImageDataGenerator(
	rotation_range=30.0,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.1,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='constant',
	vertical_flip=False)
from keras.models import load_model
datagen.fit(train_X)
#model = load_model('report4.hdf5')
print(model.summary())
N = len(train_X)

model.fit_generator(
	datagen.flow(train_X, train_Y, batch_size=128),
	epochs=200,
	validation_data=[test_X, test_Y],#[test_X ,test_Y],
	shuffle=True,		
	verbose=1,
	callbacks=[checkpoint]
	)
