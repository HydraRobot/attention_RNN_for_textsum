import tensorflow as tf
import math
import numpy as np
from helper import  *

'''
	Recurrent neural networks with bilinear products.
	Author: Povilas Daniusis, povilas.daniusis@gmail.com, 2017.
	https://github.com/povidanius/bilinear_rnn

	TODO: add dropout, normalization, initial state learning, and better initialization.
'''


class BilinearSRNN(tf.contrib.rnn.RNNCell):

    def __init__(self, input_shape, hidden_shape):
        self._num_input_rows= input_shape[0]
        self._num_input_cols = input_shape[1]
        self._num_hidden_rows = hidden_shape[0]
        self._num_hidden_cols = hidden_shape[1]
        self._num_units = hidden_shape[0] * hidden_shape[1]


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size):
        state_init = tf.Variable(tf.zeros([1, state_size()]))
        state = tf.tile(state_init, [batch_size, 1])
        return state


    def __call__(self, inputs, state, scope=None):


        H = tf.reshape(state, [-1, self._num_hidden_rows, self._num_hidden_cols])
        X = tf.reshape(inputs, [-1, self._num_input_rows, self._num_input_cols])

        with tf.variable_scope(scope or type(self).__name__):            


                    W1u = tf.get_variable("W1u",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W2u = tf.get_variable("W2u",
                        shape=[self._num_input_cols, self._num_hidden_cols])                 


                    U1u = tf.get_variable("U1u",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U2u = tf.get_variable("U2u",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])


                    Bu = tf.get_variable("Bu",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])

                 
                    H_new = tf.nn.tanh(dot(tf.transpose(dot(W1u, X), [1, 0, 2]), W2u) + dot(tf.transpose(dot(U1u, H), [1, 0, 2]), U2u) + Bu)
                    new_state = tf.reshape(H_new, [-1, self._num_hidden_rows*self._num_hidden_cols])

        return new_state, new_state



class BilinearGRU(tf.contrib.rnn.RNNCell):

    def __init__(self, input_shape, hidden_shape):
        self._num_input_rows= input_shape[0]
        self._num_input_cols = input_shape[1]
        self._num_hidden_rows = hidden_shape[0]
        self._num_hidden_cols = hidden_shape[1]
        self._num_units = hidden_shape[0] * hidden_shape[1]


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units    

    def zero_state(self, batch_size, dtype):
        state_init = tf.Variable(tf.zeros([1, self._num_units]))
        state = tf.tile(state_init, [batch_size, 1])
        return state



    def __call__(self, inputs, state, scope=None):

        batch_size = inputs.get_shape().as_list()[0]

        H = tf.reshape(state, [-1, self._num_hidden_rows, self._num_hidden_cols])
        X = tf.reshape(inputs, [-1, self._num_input_rows, self._num_input_cols])

        with tf.variable_scope(scope or type(self).__name__):            


                    W1u = tf.get_variable("W1u",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W2u = tf.get_variable("W2u",
                        shape=[self._num_input_cols, self._num_hidden_cols])


                    W1r = tf.get_variable("W1r",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W2r = tf.get_variable("W2r",
                        shape=[self._num_input_cols, self._num_hidden_cols])


                    W1h = tf.get_variable("W1h",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W2h = tf.get_variable("W2h",
                        shape=[self._num_input_cols, self._num_hidden_cols])


                    U1u = tf.get_variable("U1u",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U2u = tf.get_variable("U2u",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    U1r = tf.get_variable("U1r",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U2r = tf.get_variable("U2r",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    U1h = tf.get_variable("U1h",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U2h = tf.get_variable("U2h",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    Bu = tf.get_variable("Bu",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])

                    Br = tf.get_variable("Br",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])

                    Bh = tf.get_variable("Bh",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])


                    U = tf.nn.sigmoid( dot(tf.transpose(dot(W1u, X), [1, 0, 2]), W2u) + dot(tf.transpose(dot(U1u, H), [1, 0, 2]), U2u) + Bu)
                    R = tf.nn.sigmoid(dot(tf.transpose(dot(W1r, X), [1, 0, 2]), W2r) + dot(tf.transpose(dot(U1r, H), [1, 0, 2]), U2r) + Br)
                    H_tilde = dot(tf.transpose(dot(W1h, X), [1, 0, 2]), W2h) + R * dot(tf.transpose(dot(U1h, H), [1, 0, 2]), U2h) + Bh
  
                    H_new = U * tf.nn.tanh(H_tilde) + (tf.ones_like(U) - U) * H
                    new_state = tf.reshape(H_new, [-1, self._num_hidden_rows*self._num_hidden_cols])

        return new_state, new_state




class BilinearLSTM(tf.contrib.rnn.RNNCell):

    def __init__(self, input_shape, hidden_shape):
        self._num_input_rows= input_shape[0]
        self._num_input_cols = input_shape[1]
        self._num_hidden_rows = hidden_shape[0]
        self._num_hidden_cols = hidden_shape[1]
        self._num_units = hidden_shape[0] * hidden_shape[1]

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size):
        state_init = tf.Variable(tf.zeros([1, state_size()]))
        state = tf.tile(state_init, [batch_size, 1])
        return state


    def __call__(self, inputs, state, scope=None):
        batch_size = inputs.get_shape()[0]
        c, h = state
        C = tf.reshape(c, [-1, self._num_hidden_rows, self._num_hidden_cols])
        H = tf.reshape(h, [-1, self._num_hidden_rows, self._num_hidden_cols])
        X = tf.reshape(inputs, [-1, self._num_input_rows, self._num_input_cols])

        with tf.variable_scope(scope or type(self).__name__):            


                    W1i = tf.get_variable("W1i",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W1f = tf.get_variable("W1f",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W1o = tf.get_variable("W1o",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W1c = tf.get_variable("W1c",
                        shape=[self._num_hidden_rows, self._num_input_rows])


                    W2i = tf.get_variable("W2i",
                        shape=[self._num_input_cols, self._num_hidden_cols])

                    W2f = tf.get_variable("W2f",
                        shape=[self._num_input_cols, self._num_hidden_cols])

                    W2o = tf.get_variable("W2o",
                        shape=[self._num_input_cols, self._num_hidden_cols])


                    W2c = tf.get_variable("W2c",
                        shape=[self._num_input_cols, self._num_hidden_cols])



                    U1i = tf.get_variable("U1i",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U1f = tf.get_variable("U1f",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])


                    U1o = tf.get_variable("U1o",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U1c = tf.get_variable("U1c",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])




                    U2i = tf.get_variable("U2i",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    U2f = tf.get_variable("U2f",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    U2o = tf.get_variable("U2o",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    U2c = tf.get_variable("U2c",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])


                  
                    Bi = tf.get_variable("Bi",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])

                    Bf = tf.get_variable("Bf",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])

                    Bo = tf.get_variable("Bo",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])


                    Bc = tf.get_variable("Bc",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])


                    I = tf.nn.sigmoid(dot(tf.transpose(dot(W1i, X), [1, 0, 2]), W2i) + dot(tf.transpose(dot(U1i, H), [1, 0, 2]), U2i) + Bi)
                    F = tf.nn.sigmoid(dot(tf.transpose(dot(W1f, X), [1, 0, 2]), W2f) + dot(tf.transpose(dot(U1f, H), [1, 0, 2]), U2f) + Bf)
                    O = tf.nn.sigmoid(dot(tf.transpose(dot(W1o, X), [1, 0, 2]), W2o) + dot(tf.transpose(dot(U1o, H), [1, 0, 2]), U2o) + Bo)
                    C_tilde = tf.nn.tanh(dot(tf.transpose(dot(W1c, X), [1, 0, 2]), W2c) + dot(tf.transpose(dot(U1c, H), [1, 0, 2]), U2c) + Bc)

                    
                    C_new = I * C_tilde + F * C
                    H_new = O * tf.nn.tanh(C_new)

                    new_state = tf.contrib.rnn.LSTMStateTuple(tf.reshape(C_new, [-1, self._num_hidden_rows* self._num_hidden_cols]), tf.reshape(H_new, [-1, self._num_hidden_rows* self._num_hidden_cols]))

        return tf.reshape(H_new, [-1, self._num_hidden_rows* self._num_hidden_cols]), new_state


