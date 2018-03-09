import numpy as np
import tensorflow as tf
import data.danager as danager

dir_name = 'data/pm25_pm10/'
input_dim = 12
neuron_dim = 15
testX  = np.load(dir_name + 'test_data.npy')

trainX , trainY , valX , valY = danager.split(dir_name)

print(valX)
input_dim = trainX.shape[1]	

#exit(0)
'''
print(trainX)
print(trainY)
'''
import tensorflow as tf
x = tf.placeholder( tf.float32 , [ None , input_dim])

dev = 0.1
W = tf.Variable( tf.truncated_normal([input_dim,neuron_dim], stddev=dev))
b = tf.Variable( tf.truncated_normal([neuron_dim], stddev=dev))
h = tf.nn.relu(tf.matmul(x,W) + b)

W2= tf.Variable( tf.truncated_normal([neuron_dim,neuron_dim], stddev=dev))
b2= tf.Variable( tf.truncated_normal([1], stddev=dev))
h2 = tf.nn.relu(tf.matmul(h,W2) + b2)

W3= tf.Variable( tf.truncated_normal([neuron_dim,1], stddev=dev))
b3= tf.Variable( tf.truncated_normal([1], stddev=dev))
y = tf.nn.relu(tf.matmul(h,W3) + b3)


y_ = tf.placeholder( tf.float32 , [None, 1])

diff = tf.square( y - y_ )
loss = tf.sqrt(tf.reduce_mean(diff))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)


for _ in range(100000):

	sess.run(train_step,feed_dict={
		x  : trainX,
		y_ : trainY
	})
	if _ % 100 == 0 :
		#print('W = ',W.eval(session = sess))
		#print('b = ',b.eval(session = sess))
		loss_ = sess.run(loss, feed_dict={
			x  : trainX,
			y_ : trainY	
		})
		val_loss = sess.run(loss, feed_dict={
			x  : valX,
			y_ : valY	
		})
		print ("epoch %d loss %8g ,\tval_loss %8g" %(_,loss_ , val_loss))


out = sess.run(y, feed_dict={ x: testX } )
outtable = [["id,value"]]
for i in range(len(out)):
	outtable.append(['id_' + str(i),out[i][0]])



import csv 
cout = csv.writer(open('nn_ans.csv','w'))
cout.writerows(outtable)