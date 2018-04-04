from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from Imagegen import datagen,my_shuffle
from cnn_model import my_cnn
from pathlib import Path
import tensorflow as tf
import numpy as np
import math
import os 
import sys
######## Hyper Params ##############
validation_split = 0.1
optimizer = tf.train.AdamOptimizer(1e-3)
optimizer = tf.train.RMSPropOptimizer(1e-4)
batch_size = 128
epochs = 10000
x_dim = 48
y_dim = 7
model_save_path = 'my_cnn_model'
if not Path(model_save_path).exists():
	os.mkdir(model_save_path)
######### Data Processing ###########
image = np.load('trainX.npy')
label = np.load('trainY.npy')
train_X, test_X, train_Y, test_Y = train_test_split(image, label, test_size=validation_split, random_state=127)
print(train_X.shape , train_Y.shape)
print(test_X.shape , test_Y.shape)

#### Handling label inequiblirum #####
train_X = train_X.reshape(-1,x_dim,x_dim,1)
max_weight = int(np.max(np.sum(train_Y,axis=0)))
original_label = np.argmax(train_Y ,axis=1)

pos_weight = []
for label_type in range(7):
	num = len(train_Y[original_label == label_type])
	pos_weight.append(max_weight/num)
pos_weight= np.array(pos_weight)
print(pos_weight)
test_X = test_X.reshape(-1,x_dim,x_dim,1)
N = len(train_X)
######## PlaceHolder settings ########
x = tf.placeholder( tf.float32 , [ None , x_dim,x_dim,1 ])
y_= tf.placeholder( tf.float32 , [ None , y_dim ] )
y = my_cnn(x)
test_dict = {x: test_X , y_:test_Y}
######## Training Preprocessing #######
tensor_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y,labels =y_)
#tensor_loss = tf.nn.weighted_cross_entropy_with_logits(logits = y , targets = y_, pos_weight = pos_weight)
train_step = optimizer.minimize(tensor_loss)
tensor_pred = tf.argmax(y,1)
tensor_pred_equal = tf.equal( tensor_pred , tf.argmax(y_,1) )
tensor_acc = tf.reduce_mean( tf.cast(tensor_pred_equal , tf.float32))

######## Session Creating #############
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
observe = tf.trainable_variables()[-1]
saver = tf.train.Saver()
writer = tf.summary.FileWriter("/tmp/tensorflow/ML3", sess.graph)

####### Start Training #################
for epoch in range(epochs):
	acc , loss , siz = 0 , 0 , 0
	all_run = math.ceil(N/batch_size)
	for step , (batch_X , batch_Y) in zip(range(all_run),datagen(train_X,train_Y,batch_size)):
		fd_dict = {x:batch_X , y_:batch_Y}
		sess.run(train_step , feed_dict=fd_dict)
		now_acc,now_loss = sess.run([tensor_acc,tensor_loss] , feed_dict=fd_dict)
		acc += now_acc * len(batch_X)
		loss += np.mean(now_loss) * len(batch_X)
		siz += len(batch_X)
		if step%3 == 0:
			print('\b->' , end='')
		sys.stdout.flush()
	print('|')
	#print(np.argmax(batch_Y,axis=1)[:10])
	#print(sess.run(tensor_loss , feed_dict=fd_dict))
	val_acc,val_loss = sess.run([tensor_acc,tensor_loss] , feed_dict=test_dict)
	print("epoch %05d : acc : %.5f , loss : %.5f val_acc : %.5f , val_loss : %.5f" %\
		( epoch,acc/siz,loss/siz,val_acc,np.mean(val_loss)))
	print(sess.run([tensor_pred] , feed_dict=test_dict))
	saver.save(sess,'./' + model_save_path)
