import tensorflow as tf
import dot
import imp
imp.reload(dot)

from tensorflow.python.ops import variable_scope

"""
x 4X6 batch=4, dim=6
y 4 x 5 x 4 batch=4 len = 8 dim = 5
"""
#sess = tf.InteractiveSession()

tf.reset_default_graph()

with tf.Session() as session:
   x = tf.constant([[1,1,2,4,8,1],[1,2,4,8,1,1],[2,4,8,1,1,1],[4,8,1,1,1,2]], dtype=tf.float32)
   s=[]
   for i in range(80):
      t = random.sample({1., 1., 2., 2.}, 2) 
      s = s +  t
   y = tf.constant(s, shape=[4, 8, 5])
   init = tf.ones_initializer()
   with variable_scope.variable_scope("score") as scope:
      M, scores=dot.score(x, y, init)
   session.run(tf.global_variables_initializer())
   session.run([M,scores])

s=[]
for i in range(40):
   t = random.sample({0., 1., 2., 4.}, 4)
   s = s +  t
y = tf.constant(s, shape=[4, 8, 5])
