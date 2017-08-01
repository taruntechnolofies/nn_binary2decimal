#This program trains and verifys a  NN for Binary to decimal Conversion

import numpy as np
import tensorflow as tf

#----------------------------------global variables------------------------------------

#number of bits in binary, also the inputs to the nn
# if you increase the bits the, this training part is not very good for that;
# because number of samples increases exponentially, and lot of time is waisted in copying data
#but if you use pipeline or constant instead of feed_dic, the train time can be reduced significantly
n_inputs = 8
#number of neuraon in layers
n_layer1 = 6
n_outputLayer = 1
learningrate = 0.01
train_iterations = 7000
cost_print_itr = 100
train_data_percent = 0.8

#--------------------------------data generation part----------------------------------
dataRange = np.power(2,n_inputs)

#generates all binary in the given bit size
#also genrates the corresponding decimal output
#bundle the input and output in a touple
#and add then to list
def generate_bin_data():
	data = []
	for i in range(dataRange):
		a = [int(x) for x in list('{0:0b}'.format(i))]
		zer = n_inputs - len(a)
		arr = [0]*zer
		a = arr + a
		data.append((a,[i]))
	return data

#generate data from function
d = generate_bin_data()

#shuffle the data randomly
np.random.shuffle(d)

# saperate the test data and train data
n_train = int(np.rint(dataRange*train_data_percent))
n_test = dataRange-n_train
train = d[0:n_train]
test = d[n_train:dataRange]

#function to split the input output from touple
def split_inout(data):
	inputs = []
	outputs = []
	for d in data:
		inputs.append(d[0])
		outputs.append(d[1])
	return inputs,outputs

#spliting the data into input and output
train_inp, train_out = split_inout(train)
test_inp, test_out = split_inout(test)

#----------------------------------------NN Model Creation Part-------------------------------

#function to add layers in neural network
def add_layer(node_n,input_data,input_size,actvation_function):
	W = tf.Variable(tf.random_normal(shape=[input_size,node_n]))
	B = tf.Variable(tf.random_normal(shape=[node_n]))
	val = tf.add(tf.matmul(input_data,W),B)
	if actvation_function==None:
		return val
	else:
		return actvation_function(val)

#placeholders for inputs and outputs
x_plac = tf.placeholder(tf.float32,shape=[None,n_inputs])
y_plac = tf.placeholder(tf.float32,shape=[None,n_outputLayer])

#layer 1
l1 = add_layer(n_layer1,x_plac,n_inputs,tf.nn.relu)
#Output layer
outlayer = add_layer(n_outputLayer,l1,n_layer1,None)

error = tf.pow(y_plac-outlayer,2)/2
cost = tf.reduce_mean(error)

optimiser = tf.train.AdamOptimizer(learningrate)
train_step = optimiser.minimize(cost)

init = tf.global_variables_initializer()

#Traing the Model
with tf.Session() as sess:
	sess.run(init)
	for i in range(train_iterations):
		sess.run(train_step,feed_dict={x_plac:train_inp,y_plac:train_out})
		if i%cost_print_itr == 0:
			print(sess.run(cost,feed_dict={x_plac:train_inp,y_plac:train_out}))
	resultData = sess.run(outlayer,feed_dict={x_plac:test_inp})
	sess.close()
#---------------------------------------------------Verification Part-------------------------

# printing the comparison
print("-----------------------Compare Result----------------------------")
i = 0
for data in resultData:
	pred = data[0]
	real = test_out[i][0]
	err = ((real-pred)/real)*100
	print("Predicted: ", pred , " Real:", real, "error: ", err, "%" )
	i = i+1
