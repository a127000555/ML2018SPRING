import math
import argparse
import numpy as np
import pandas as pd 
import tensorflow as tf

def init_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def CNN(x, size, label):
	'''
	input -> CNN -> CNN -> CNN -> Flatten -> Dense * 2 -> output
	'''
	x_image = tf.reshape(x, [-1, size, size, 1])
	### CNN 1 ####
	w_conv1 = init_weights([5,5,1,32])
	b_conv1 = init_weights([32])
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### CNN 2 ####
	w_conv2 = init_weights([3,3,32,32])
	b_conv2 = init_weights([32])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	### CNN 3 ###
	w_conv3 = init_weights([3,3,32,32])
	b_conv3 = init_weights([32])
	h_conv3 = tf.nn.relu(conv2d(h_pool2,w_conv3) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3)	

	#### Flatten layer ####
	h_flat = tf.reshape(h_pool3, [ -1 , size//8 * size//8 * 32 ])

	#### dense1 ####
	w_fc1 = init_weights([ size//8 * size//8 * 32 , 32])
	b_fc1 = init_weights([32])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dnense2 ####
	W_fc2 = init_weights([32, 32])
	b_fc2 = init_weights([32])
	h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

	#### output ####
	W_fc3 = init_weights([32, label])
	b_fc3 = init_weights([label])
	y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
	
	return y_conv

def read_data(training, testing):
	train = pd.read_csv(training)
	test = pd.read_csv(testing)
	ret = {
		'trainX': np.array(train['feature'].str.split(" ").values.tolist()).reshape(-1, 48*48).astype(np.float),
		'trainY': pd.get_dummies(train['label']).values.astype(int),
		'test_X': np.array(test['feature'].str.split(" ").values.tolist()).reshape(-1, 48*48).astype(np.float),
	}
	return ret

def next_batch(batch_size):
	rand = np.random.permutation(data['trainX'].shape[0])
	rand = rand[:batch_size]
	return data['trainX'][rand], data['trainY'][rand]

parser = argparse.ArgumentParser(description='ML HW3')    
parser.add_argument('train', type=str, help='training data')
parser.add_argument('test', type=str, help='testing data')
parser.add_argument('model', type=str, help='save model path')
# parser.add_argument('out', type=str, help='output path')
parser.add_argument('-s','--scale', type=bool, help='scale data', default=True) 
args = parser.parse_args()

data = read_data(args.train, args.test)	# dictionary (trainX, trainY, test_X) -> three numpy array
print ("finish parsing")
# if args.scale is True:
# 	data['trainX'], data['test_X'] = scale()
learning_rate = 1e-4
x = tf.placeholder( tf.float32 , [ None , 48*48 ])	# training x
y_= tf.placeholder( tf.float32 , [ None , 7 ])	# training y
y = CNN(x, 48, 7)	# fit y
print ("model is ok")

#### model compile ####
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_pred = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
acc = tf.reduce_mean( tf.cast(correct_pred , tf.float32))

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

#### Traingin ####
for i in range(1000):
	batch = next_batch(100)
	train_step.run(session = sess , feed_dict={
			x : batch[0],
			y_: batch[1]
		})
	train_acc = sess.run(acc,feed_dict={
		x : batch[0],
		y_: batch[1],
	})
	if i % 100 == 0:
		print("epoch = ", i, "Acc = ", train_acc)
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, args.model)
