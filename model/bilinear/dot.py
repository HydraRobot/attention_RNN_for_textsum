import tensorflow as tf
import numpy as np
import math

def is_sparse(tensor):
    return isinstance(tensor, tf.SparseTensor)

def int_shape(x):
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])


def ndim(x):
    if is_sparse(x):
        return x._dims
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a ND tensor
    with a ND tensor, it reproduces the Theano behavior.
    (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))
    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.
    # Returns
        A tensor, dot product of `x` and `y`.
    # Examples
    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(2, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
    ```
    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(32, 28, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
    ```
    ```python
        # Theano-like behavior example
        >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
        >>> y = K.ones((4, 3, 5))
        >>> xy = K.dot(x, y)
        >>> K.int_shape(xy)
        (2, 4, 5)
    ```
    """
    if hasattr(tf, 'unstack'):
        unstack = tf.unstack
    else:
        unstack = tf.unpack
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(int_shape(x), unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(int_shape(y), unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if is_sparse(x):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)

    return out

def general_attention(hd,hs, init):
   """this function is calculate attetion of hd over hs.
      Argus:
      hd: hidden_state [batch_size X d_dim]
      hs: source [batch_size X len X s_dim]
      init: Wattn initialization way
   
      output: 
      attn_dists: attention distribution [batch_size X len ]  , scalar  
   """
   hd_shape = hd.get_shape()
   hs_shape = hs.get_shape() 

   Wattn = tf.get_variable("Wattn", [hd_shape[1].value, hs_shape[2].value], dtype=tf.float32, initializer = init)
    
   M=tf.expand_dims(dot(hd, Wattn), axis=1)
   attn_dists = tf.nn.softmax(tf.reduce_sum((hs * M), 2))
   context = tf.reduce_sum(hs * attn_dists, axis=1)
   return attn_dists, context 
