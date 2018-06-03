from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from bilinear_rnn import *
import numpy as np
from helper import  *
import matplotlib.pyplot as plt

# dim of input matrices
dx1 = 10
dx2 = 10
# dim of hidden state
dh1 = 30 
dh2 = 30 
#dim of output matrices
dy1 = 10
dy2 = 10

training_size = 2000
num_epochs = 200
learning_rate = 0.001
batch_size = 128
n_steps = 100
logs_path = '/tmp/tensorflow_logs/example'


def imshow(x):
	plt.imshow(x,cmap='gray')
	plt.show()

def imshow2(x,y):
	fig = plt.figure()
	ax = fig.add_subplot(2, 1, 1)
	ax.imshow(x,cmap='gray')
	ax.autoscale(False)
	ax2 = fig.add_subplot(2, 1, 2, sharex=ax, sharey=ax)
	ax2.imshow(y,cmap='gray')
	ax2.autoscale(False)
	ax.set_adjustable('box-forced')
	ax2.set_adjustable('box-forced')
	plt.show()

def normalize(x):
	return (x - mean(x, axis=0)) / std(A, axis=0)
	

def count_trainable_parameters():
 total_parameters = 0
 for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:      
        variable_parametes *= dim.value    
    total_parameters += variable_parametes
 return total_parameters

# "structure"
# first row - Gaussian
# remaining ones - multiplies of first row 
def get_sample():
	x = np.random.normal(loc = 0.0, scale = 1.0, size=(dx1,dx2))
	for i in range(dx1-1):
		x[i+1,:] = x[0,:] * np.random.normal(loc = 0.0, scale = 1.0)
	return x
# "no structure" - i.i.d. Gaussian entries.

def get_sample0():
	x = np.random.normal(loc = 0.0, scale = 1.0, size=(dx1,dx2))
	return x

def generate_data(num_samples = 2000):
	data_x = np.zeros((num_samples, n_steps, dx1, dx2))
	data_y = np.zeros((num_samples, dy1, dy2))

	for m in range(num_samples):

	  for n in range(n_steps):
	  	data_x[m, n,:,:] =  get_sample()

	  #data_y[m, :, :] = np.maximum(data_x[m,1,:,:], 0.5 * (data_x[m, 18,:,:] + data_x[m, 19,:,:]))
	  data_y[m, :, :] =  0.5 * (data_x[m,-70,:,:] + data_x[m, -1,:,:])
	return data_x, data_y
			
def get_parameters():
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        print(k, v)		
	
	
x = tf.placeholder("float", [None, n_steps, dx1, dx2])
y = tf.placeholder("float", [None, dy1, dy2])


weights = {
    'left': tf.Variable(tf.random_normal([dy1,dh1])),    
    'right': tf.Variable(tf.random_normal([dh2,dy2])),    
    'biases': tf.Variable(tf.random_normal([dy1, dy2]))
}


'''
def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.pack([batch_size] + [1] * len(shape)))
        var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        return var

    return variable_state_initializer
'''

def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)

def get_state_reset_op(state_variables, cell, batch_size):
    # Return an operation to set each variable in a list of LSTMStateTuples to zero
    zero_states = cell.zero_state(batch_size, tf.float32)
    return get_state_update_op(state_variables, zero_states)

def RNN(x, weights):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input1, n_input2)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input1, n_input2)
	
   #with tf.variable_scope('RNN_MODEL'):
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2, 3])
    # Reshaping to (n_steps*batch_size, n_input1*n_input2)
    x = tf.reshape(x, [-1, dx1*dx2])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input1*n_input2)
    x = tf.split(x, n_steps, 0)

    rnn_cell = BilinearGRU(input_shape = [dx1,dx2], hidden_shape = [dh1, dh2])
    #rnn_cell = BilinearSRNN(input_shape = [dx1,dx2], hidden_shape = [dh1, dh2])
    #rnn_cell = BilinearLSTM(input_shape = [dx1,dx2], hidden_shape = [dh1, dh2])
    #rnn_cell = rnn.GRUCell(dh1*dh2)

    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    #all_states = get_state_variables(batch_size, rnn_cell)

    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    out = tf.reshape(outputs[-1],[-1, dh1, dh2])

    # slow!	
    prediction = dot(tf.transpose(dot(weights['left'], out), [1, 0, 2]), weights['right']) + weights['biases']
    #prediction = out

    return prediction


X_train,Y_train = generate_data(training_size)
X_test, Y_test = generate_data(3000)


with tf.name_scope('Model'):
    pred = RNN(x, weights)
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(tf.subtract(pred, y))) 
with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


init = tf.global_variables_initializer()

tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merge_all()




with tf.Session() as sess:

    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for epoch in range(num_epochs):

        start = 0
        end = batch_size
        batchloss = 0.0
        num_batches = int(training_size/batch_size) 

        for i in range( num_batches ):

            batch_x = X_train[start:end,...]
            batch_y = Y_train[start:end,...]

            #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            _, c, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={x: batch_x, y: batch_y})
	    batchloss = batchloss + sess.run(loss, feed_dict={x: batch_x, y: batch_y})
	    summary_writer.add_summary(summary, epoch*num_batches + i)

            start = end
            end = start + batch_size

	testloss = sess.run(loss, feed_dict={x: X_test, y: Y_test})
	#print("Epoch {} train loss {}, testloss {}".format(epoch, batchloss / int(1024/batch_size), testloss))
	print("{}\t{}\t{}".format(epoch, batchloss / int(training_size/batch_size), testloss))
	predictions = sess.run(pred, feed_dict={x: X_test})

    imshow2(Y_test[1,...] , predictions[1,...])

    print(get_parameters())



    print("Trainable parameters {}". format(count_trainable_parameters()))
    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")


