from keras.models import load_model
from keras.models import Model
import numpy as np
import csv
import sys


trainfile = 'train.csv'
testfile  = 'test.csv'
outfile = 'output.csv'
if len(sys.argv) > 1:
	trainfile = sys.argv[1]
	testfile = sys.argv[2]
	outfile = sys.argv[3]
 
train = [ row for row in csv.reader(open(trainfile,'r'))][1:]
test =  [ row for row in csv.reader(open(testfile,'r'))][1:]

image = []
label = []
test_image = []
for row in train:
	l = np.zeros((7))
	l[int(row[0])] = 1
	image.append(row[1].split())
	label.append(l)
for row in test:
	test_image.append(row[1].split())
	#exit()

image = np.array(image).astype(np.float).reshape(-1,48,48,1)
label = np.array(label).astype(np.float)
test = np.array(test_image).astype(np.float).reshape(-1,48,48,1)
print(test.shape)

model = load_model('MA.h5')
print(model.summary())


mean = np.mean(image.reshape(-1,48,48,1),axis=0)
std = np.std(image.reshape(-1,48,48,1),axis=0)
print(mean.shape , std.shape , test.shape)
test = (test - mean) / std
output = model.predict(test , verbose=1)

fout = open(outfile,'w')

fout.write("id,label\n")
_ = 0
for row in output:
	fout.write('%d,%d\n'%(_,np.argmax(row)))
	_+=1
