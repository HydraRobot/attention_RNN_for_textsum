# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file defines the decoder"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell, initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None):
  """
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
    enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
    cell: rnn_cell.RNNCell defining the cell function and size.
    initial_state_attention:
      Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
    pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
    use_coverage: boolean. If True, use coverage mechanism.
    prev_coverage:
      If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not None in decode mode when using coverage.

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of
      shape [batch_size x cell.output_size]. The output vectors.
    state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
    attn_dists: A list containing tensors of shape (batch_size,attn_length).
      The attention distributions for each decoder step.
    p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
    coverage: Coverage vector on the last step computed. None if use_coverage=False.
  """
  with variable_scope.variable_scope("attention_decoder") as scope:
    #tf.logging.info("####encoder_states", encoder_states) 
    batch_size = encoder_states.get_shape()[0].value # if this line fails, it's because the batch size isn't defined
    attn_size = encoder_states.get_shape()[2].value # if this line fails, it's because the attention length isn't defined

    """not need ecoder_feature anymore by Margarite
    # Reshape encoder_states (need to insert a dim)
    encoder_states_t = tf.expand_dims(encoder_states, axis=2) # now is shape (batch_size, attn_len, 1, attn_size)
    encoder_state1 = encoder_states
    encoder_state1 = tf.Print(encoder_state1, [encoder_state1], "meassage=encoder_state1 shape", summarize=10000)
    encoder_states = encoder_states_t
    """ 


    """don't need attention_vec_size , call dot.general_attention instead by Margarite 
    # To calculate attention, we calculate
    #   v^T tanh(W_h h_i + W_s s_t + b_attn)
    # where h_i is an encoder state, and s_t a decoder state.
    # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
    # We set it to be equal to the size of the encoder states.
    attention_vec_size = attn_size
    """

    """
    # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
    W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
    encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)
    """

    """ don't need coverage. use  decoder attention instead. by Margarite
    # Get the weight vectors v and w_c (w_c is for coverage)
    v = variable_scope.get_variable("v", [attention_vec_size])
    if use_coverage:
      with variable_scope.variable_scope("coverage"):
        w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

    if prev_coverage is not None: # for beam search mode with coverage
      # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
      prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage,2),3)
    """

    outputs = []
    attn_dists = []
    decoder_attn_dists = []  
    p_gens = []
    state = initial_state
    decoder_states = []
    init = tf.ones_initialization() 

    #coverage = prev_coverage # initialize coverage to None or whatever was passed in
    initial_context_vector = array_ops.zeros([batch_size, hidden_state_size]) #initial_context_vector is  decoder context vector of t=0
    context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.

    """ in Luong mode. attention is calculated after LSTM(). This should be put in while.... by Margarita
    if initial_state_attention: # true in decode mode
      # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
      context_vector, _, coverage = attention(initial_state, coverage) # in decode mode, this is what updates the coverage vector
    """
    
    for i, inp in enumerate(decoder_inputs):
      tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      """ don't merge input with context in Luong style attention  
      # Merge input and previous attentions into one vector x of the same size as inp
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + [context_vector], input_size, True)
      """

      # 1. Run the decoder RNN cell. cell_output = decoder state
      cell_output, state = cell(inp, state)

      # 2. Run the attention mechanism.
      if i == 0 and initial_state_attention:  # always true in decode mode
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
          context_vector, attn_dist, _ = attention(state, coverage) # don't allow coverage to update
      else:
          decoder_attn_dist, decoder_context  = general_attention(state, decoder_states, init)#decoder_attn_dist:  attention dist on previouse decoder states; decoder_context: context generated from weighted previous  decoder state. 
 
      decoder_attn_dists.append(decoder_attn_dist)
      decoder_states.append(state)

      # 3. Run attention mechanism
 
      if i == 0 and initial_state_attention:  # always true in decode mode
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
          context_vector, attn_dist, _ = attention(state, coverage) # don't allow coverage to update
      else:
          attn_dist, context  = general_attention(state, encoder_states, init)#attn_dist:  attention dist on previouse encoder states; context: context generated from weighted previous  encoder state. 
 
      attn_dists.append(attn_dist)

      4.  # Calculate p_gen TODO
      #TODO: p(ut=1)=sigmoid(Wu[htd||Cte||Ctd]+bu)
      if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
          p_gen = linear([context_vector, state.c, state.h, x], 1, True) # a scalar
          p_gen = tf.sigmoid(p_gen)
          p_gens.append(p_gen)

      5. # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer TODO
      # This is V[s_t, h*_t] + b in the paper
      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + [context_vector] + [decoder_context_vector], cell.output_size, True)
      outputs.append(output)

    """remove coverage"""
    # If using coverage, reshape it
    if coverage is not None:
      coverage = array_ops.reshape(coverage, [batch_size, -1])
    """
    return outputs, state, attn_dists, decoder_attn_dists, p_gens, decoder_states 



def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    tf.logging.info("total_arg_size %i", total_arg_size)
    tf.logging.info("output_size %i", output_size)
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term


def bilinear(input1, input2, scope=None):
   """bilinear function for general attention score: s=htwhe
   Args: input1: ht, current decoder hidden state, 2D tensor, b_s X state_size
         input2: he, encoder hidden state at all step, 3D tensor, b_s X attn_length X attn_size]
         scope: variable scope
   Returns:
         score b_s X attn_length
   """
   decoder_state = input1
   state = input2
   #state_size = decoder_state.get_shape[1].value
   shapes = state.get_shape().as_list()  
   attn_size = shapes[2] 
   attn_length = tf.shape(state)[1]
   tf.logging.info("attn_size:%d", attn_size)
   attn_length = tf.Print(attn_length, [attn_length], "message=attn_length")
   #W = tf.get_variables(Wattn, [state_size, attn_size]
   with tf.variable_scope(scope or "Bilinear"):
       hw=linear(decoder_state, attn_size, True)
   hw=tf.expand_dims(hw, 1)
   state = tf.transpose(state, perm=[0,2,1])
   score = tf.matmul(hw, state)
   score = tf.reshape(score, [-1,attn_length])
   return score
