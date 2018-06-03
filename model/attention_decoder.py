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
    batch_size = encoder_states.get_shape()[0].value # if this line fails, it's because the batch size isn't defined
    attn_size = encoder_states.get_shape()[2].value # if this line fails, it's because the attention length isn't defined

    # Reshape encoder_states (need to insert a dim)
    encoder_states = tf.expand_dims(encoder_states, axis=2) # now is shape (batch_size, attn_len, 1, attn_size)

    # To calculate attention, we calculate
    #   v^T tanh(W_h h_i + W_s s_t + b_attn)
    # where h_i is an encoder state, and s_t a decoder state.
    # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
    # We set it to be equal to the size of the encoder states.
    attention_vec_size = attn_size

    # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
    W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
    encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)

    # Get the weight vectors v and w_c (w_c is for coverage)
    v = variable_scope.get_variable("v", [attention_vec_size])
    if use_coverage:
      with variable_scope.variable_scope("coverage"):
        w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

    if prev_coverage is not None: # for beam search mode with coverage
      # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
      prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage,2),3)

    def attention(decoder_state, coverage=None):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        decoder_state: state of the decoder
        coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

      Returns:
        context_vector: weighted sum of encoder_states
        attn_dist: attention distribution
        coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
      """
      with variable_scope.variable_scope("Attention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        tf.logging.debug("i m here, attention_vec_size", attention_vec_size)
        decoder_features = linear(decoder_state, attention_vec_size, False) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)

        def masked_attention(e):
          """Take softmax of e then apply enc_padding_mask and re-normalize"""
          attn_dist = nn_ops.softmax(e) # take softmax. shape (batch_size, attn_length)
          attn_dist *= enc_padding_mask # apply mask
          masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
          return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize

        if use_coverage and coverage is not None: # non-first step of coverage
          # Multiply coverage vector by w_c to get coverage_features.
          coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features + coverage_features), [2, 3])  # shape (batch_size,attn_length)

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          # Update coverage vector
          coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3]) # calculate e

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          if use_coverage: # first step of training
            coverage = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage

        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage


    def intra_attention(states, coverage=None):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        states:tf decoder states. [batch_size, prev_decoder_length, cell.state_size]  
        coverage: latest: abandoned. Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

      Returns:
        context_vector: weighted sum of states[0:-1]
        attn_dist: attention distribution
        coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
          V^Ttanh(Whhi + WsS + b_attn) 
      """
      with variable_scope.variable_scope("intra_attention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        tf.logging.debug("i m here, attention_vec_size", attention_vec_size)
        #(1) state by Mag...
        #state = states[:,-1,:]
        #(2) reshpae states by Mag...
        states = tf.expand_dims(states, axis=2) # now is shape (batch_size, decoder_state_len, 1, cell.state_size)
        decoder_features = linear(state, attention_vec_size, False) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
        #(3) intra_decoder_features by Mag...
        intra_decoder_len = tf.shape(states)[1]
        W_d = variable_scope.get_variable("W_d", [1, 1, hidden_num*2, attention_vec_size]) 
        intra_decoder_features = nn_ops.conv2d(states, W_d, [1, 1, 1, 1], "SAME") # shape (batch_size,decoder_state_length,1,attention_vec_size)
        def masked_attention(e):
          """Take softmax of e then apply enc_padding_mask and re-normalize"""
          attn_dist = nn_ops.softmax(e) # take softmax. shape (batch_size, attn_length)
          ad1 = tf.shape(attn_dist)[1] 
          tf.logging.info("***attn_dist shape %i", ad1)
          epm1 = tf.shape(enc_padding_mask)[1]
          tf.logging.info("```enc_padding_mask %i", epm1)
          #attn_dist *= enc_padding_mask # apply mask
          #print("***attn_dist", tf.shape(attn_dist))
          masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
          return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize

        if use_coverage and coverage is not None: # non-first step of coverage
          # Multiply coverage vector by w_c to get coverage_features.
          coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(intra_decoder_features + decoder_features + coverage_features), [2, 3])  # shape (batch_size,attn_length)

          # Calculate attention distribution
          attn_dist = masked_attention(e)
          # Update coverage vector
          coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          # Calculate v^T tanh(W_h h_i + W_s S_t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(intra_decoder_features + decoder_features), [2, 3]) # calculate e

          # Calculate attention distribution
          attn_dist = masked_attention(e)
          if use_coverage: # first step of training
            coverage = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage

        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * states, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage
    
    outputs = []
    attn_dists = []
    intra_attn_dists = [] 
    p_gens = []
    
    states = tf.expand_dims(tf.concat(cell.zero_state(batch_size, tf.float32), axis=1), 1) 
    state = initial_state
    hidden_num = initial_state[0].get_shape()[1].value
    tf.logging.info("hidden_num", hidden_num)
    coverage = prev_coverage # initialize coverage to None or whatever was passed in
    context_vector = array_ops.zeros([batch_size, attn_size])
    context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    intra_context_vector = array_ops.zeros([batch_size, attn_size])
    intra_context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    if initial_state_attention: # true in decode mode
      # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
      context_vector, _, coverage = attention(initial_state, coverage) # in decode mode, this is what updates the coverage vector
      intra_context_vector, intra_attn_dist, _ = intra_attention(states)
    for i, inp in enumerate(decoder_inputs):
      print("i:", i)
      tf.logging.debug("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      # Merge input and previous attentions into one vector x of the same size as inp
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      #(1)add intra_context_vector by Mag... 
      x = linear([inp] + [context_vector] +  [intra_context_vector], input_size, True)

      # Run the decoder RNN cell. cell_output = decoder state
      
      
      cell_output, state = cell(x, state)

      

      # Run the attention mechanism.
      # added intra attention intra_decoder_vector, intra_attn_dist, s by Mag...
      if i == 0 and initial_state_attention:  # always true in decode mode
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
          context_vector, attn_dist, _ = attention(state, coverage) # don't allow coverage to update
          #(4) call intra_attention fro intra_context_vector, intra_attn_dist by Mag...
          intra_context_vector, intra_attn_dist, _ = intra_attention(states)  
      else:
        context_vector, attn_dist, coverage = attention(state, coverage)
        #(4) call intra_attention fro intra_context_vector, intra_attn_dist by Mag...
        intra_context_vector, intra_attn_dist, _ = intra_attention(states)  
      attn_dists.append(attn_dist)
      intra_attn_dists.append(intra_attn_dists)
      #(5) add state to states by Mag...
      #concat state.c and state.h 
      state_s = tf.expand_dims(tf.concat(state, axis=1), 1) #[b_s, dims*2] -> [b_s, 1, dims]
      states = tf.concat([states, state_s], 1)
      #states.append(state) 

      # Calculate p_gen
      if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
          p_gen = linear([context_vector, state.c, state.h, x], 1, True) # a scalar
          #p_gen = tf.Print(p_gen, [p_gen], "message=before sigmoid, p_gen:", summarize=10000) 
          p_gen = tf.sigmoid(p_gen)
          #p_gen = tf.Print(p_gen, [p_gen], "message=after sigmoid, p_gen:", summarize=10000) 
          p_gens.append(p_gen)

      # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
      # This is V[s_t, h*_t] + b in the paper
      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + [context_vector], cell.output_size, True)
      outputs.append(output)

    # If using coverage, reshape it
    if coverage is not None:
      coverage = array_ops.reshape(coverage, [batch_size, -1])

    return outputs, state, attn_dists, p_gens, coverage



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
  tf.logging.debug("output_size %i", output_size)
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
    #tf.logging.info("total_arg_size %i", total_arg_size)
    """matrix is the weights matrix"""
    #tf.logging.info("create matrix ")
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    """rewrite the matrix to known matrix for test"""
    #matrix = tf.get_variable("Matrix", [total_arg_size, output_size], initializer=tf.ones_initializer())
    """im = tf.ones([total_arg_size], tf.float32)
    for o in range(output_size-1):
       new =im[-total_arg_size:] + 1                
       im = tf.concat([im, new], 0)
    im = tf.transpose(tf.reshape(im, [output_size,total_arg_size]))
    tf.logging.debug("im size%s", im.get_shape()) 
    matrix = tf.get_variable("Matrix", initializer=im)
    #matrix = tf.Print(matrix, [matrix], "message=[deeper and deeper]matrix is", summarize=100)
    """

    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
      tf.logging.debug("debug_matrix", matrix)
      tf.logging.debug("debug:", tf.concat(axis=1, values=args)) 
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    tf.logging.debug("bias_term %s",bias_term)
  return res + bias_term

