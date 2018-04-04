import tensorflow as tf
# He normal
#initializer = tf.keras.initializers.he_normal()

# Glorot normal
initializer = tf.contrib.layers.xavier_initializer() 
def cret(shape):
	return tf.Variable(initializer(shape))

def uniform_cret(shape):
	# uniform distribution 0,1
	return tf.Variable(tf.random_uniform(shape))

def truncate_cret(shape):
	# truncate normal distribution 0.1
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def batch_norm(x):
	return tf.layers.batch_normalization (x)

def conv(x,w):
	return tf.nn.conv2d(x,w , strides=[1, 1, 1, 1], padding = 'SAME')

def pool(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	
def simple_conv(inputs,filters,filters_size):
	print('inputs: ',inputs.shape)
	print('cnn_conv: ',[ filters_size , filters_size , int(inputs.shape[3]) , filters ])
	w = cret( [ filters_size , filters_size , int(inputs.shape[3]) , filters ])
	b = cret( [ filters ])
	return tf.add(conv(inputs,w), b)

def drop(x,rate):
	return tf.nn.dropout(x,1-rate)

def fully_connected(inputs,neurons):
	w = cret( [  int(inputs.shape[1]) , neurons])
	b = cret( [ neurons ])
	return tf.add(tf.matmul(inputs,w),b)


def my_cnn(inputs):
	outputs = tf.reshape( inputs , [ -1 , 48 , 48 , 1 ])
	
	############### CNN HyperParameters ##################
	cnn_filters		= 	[ 	64,		128,	512,	512 ]#
	cnn_droprate	=	[ 	0.25,	0.3,	0.35,	0.4	]#
	cnn_filter_size	=	[	5,		3,		3,		3	]#
	######################################################

	for filters,filter_size,droprate in zip(cnn_filters,cnn_filter_size,cnn_droprate):
		outputs = simple_conv(outputs,filters,filter_size)
		outputs = tf.nn.leaky_relu(outputs , 1./20)
		#outputs = batch_norm(outputs)
		outputs = pool(outputs)
		#outputs = drop(outputs,droprate)
	
	#############  flatten ##############
	outputs = tf.layers.flatten(outputs)#
	#####################################
	outputs = tf.reshape(inputs , [-1,48*48])
	######## DNN HyperParmaters ##################
	dnn_neurons 	= 	[	2048,2048,2048	]		
	dnn_droprate	=	[	0.1,	0.1,	0.1]	
	##############################################

	#for neurons,droprate in zip(dnn_neurons,dnn_droprate):
	outputs = fully_connected(outputs,512)
	outputs = tf.nn.relu(outputs)
	outputs = fully_connected(outputs,512)
	outputs = tf.nn.relu(outputs)
	#outputs = batch_norm(outputs)
	#outputs = drop(outputs,droprate)


	outputs = fully_connected(outputs,7)

	return outputs

	
