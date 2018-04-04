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
validation_split = 0.15
train_X, test_X, train_Y, test_Y = train_test_split(image, label, test_size=validation_split, random_state=127)

### MACRO ###
def conv(filters , size , strides = (1,1)):
	return Conv2D(filters, strides = strides,kernel_size=(size, size),padding='same', kernel_initializer='glorot_normal')

def pool():
	return MaxPooling2D(pool_size=(2, 2), padding='same')

def den(neurons):
	return Dense(neurons, activation='relu', kernel_initializer='glorot_normal')

def simple_cnn(inputs , filters , filters_size , strides = (1,1 ), droprate = None):
	if droprate is None:
		return BatchNormalization()(LeakyReLU(alpha=1./20)(conv(filters,filters_size,strides)(inputs)))
	return Dropout(droprate)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(filters,filters_size,strides)(inputs))))

def res(inputs, filters , filters_size, droprate = None):
	if droprate is None:
		conv1 = BatchNormalization()(LeakyReLU(alpha=1./20)(conv(filters,filters_size)(inputs)))
		conv2 = BatchNormalization()(LeakyReLU(alpha=1./20)(conv(filters,filters_size)(conv1)))
		return keras.layers.Add()([conv2 , inputs])
	else:
		conv1 = Dropout(droprate)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(filters,filters_size)(inputs))))
		conv2 = Dropout(droprate)(BatchNormalization()(LeakyReLU(alpha=1./20)(conv(filters,filters_size)(conv1 ))))
		return keras.layers.Add()([conv2 , inputs])

def hard_res(inputs, filters , droprate = None):
	if droprate is None:
		conv1 = simple_cnn(inputs , filters , 3 , (2,2))
		conv2 = simple_cnn(conv1 , filters , 3) # 64 * 24*24
		conv_1_5 = simple_cnn(inputs , filters , 3 , (2,2))
	else:
		conv1 = Dropout(droprate)(simple_cnn(inputs , filters , 3 , (2,2)))
		conv2 = Dropout(droprate)(simple_cnn(conv1 , filters , 3))
		conv_1_5 = Dropout(droprate)(simple_cnn(inputs , filters , 3 , (2,2)))
	return keras.layers.Add()([conv_1_5 , conv2])
	
### MODEL ###
x = Input(shape=(48,48,1))

output = res(x , 64 , 5 ,0.3)
output = res(output , 64 , 5,0.3)	# 32 * 48*48
output = res(output , 64 , 5,0.3)	# 32 * 48*48
output = hard_res(output , 128,0.3)
output = res(output , 128 , 3,0.3)
output = res(output , 128 , 3,0.3)
output = res(output , 128 , 3,0.3)

output2 = simple_cnn(output , 256 , 3 , (2,2),0.35)
output2 = simple_cnn(output2 , 256 , 3, (1,1) ,0.35) # 64 * 24*24
output = simple_cnn(output , 256 , 3 , (2,2),0.35)
output =  keras.layers.Add()([output , output2])

output = res(output , 256 , 3,0.4)
output = res(output , 256 , 3,0.4)
output = res(output , 256 , 3,0.4)


output2 = simple_cnn(output , 512 , 3 , (2,2),0.45)
output2 = simple_cnn(output2 , 512, 3,(1,1) ,0.45) # 64 * 24*24
output = simple_cnn(output , 512 , 3 , (2,2),0.45)
output =  keras.layers.Add()([output , output2])

output = res(output , 512 , 3,0.5)
output = res(output , 512 , 3,0.5)

output2 = simple_cnn(output , 512 , 3 , (2,2),0.5)
output2 = simple_cnn(output2 , 512, 3,(1,1) ,0.5) # 64 * 24*24
output = simple_cnn(output , 512 , 3 , (2,2),0.5)
output =  keras.layers.Add()([output , output2])


flat = Flatten()(output)

dense = Dropout(0.5)(BatchNormalization()(den(1024)(flat)))
dense = Dropout(0.5)(BatchNormalization()(den(1024)(dense)))
y = Dense(7, activation='softmax', kernel_initializer='glorot_normal')(dense)


model = Model(x,y)
#print(model.summary())
### COMPILE ###
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('super_resnet_model.hdf5', monitor='val_acc', save_best_only=True, verbose=1)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip = True,
    channel_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2
)

from keras.models import load_model
datagen.fit(train_X)
model = load_model('super_resnet_model.hdf5')
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



model = load_model('super_resnet_model.hdf5')
output = model.predict(np.load('testX.npy').reshape(-1,48,48,1))

fout = open('ans.csv','w')

fout.write("id,label\n")
_ = 0
for row in output:
	fout.write('%d,%d\n'%(_,np.argmax(row)))
	_+=1