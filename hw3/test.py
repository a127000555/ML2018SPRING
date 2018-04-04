from keras.models import load_model
from keras.models import Model
import numpy as np
image = np.load('trainX.npy').reshape(-1,48,48,1)
label = np.load('trainY.npy')
model = load_model('super_resnet_model.hdf5')
print(model.summary())
training_res = model.evaluate(image,label)
print('training loss: %g , acc: %g'%(training_res[0] , training_res[1]) )
output = model.predict(np.load('testX.npy').reshape(-1,48,48,1))

fout = open('ans.csv','w')

fout.write("id,label\n")
_ = 0
for row in output:
	fout.write('%d,%d\n'%(_,np.argmax(row)))
	_+=1