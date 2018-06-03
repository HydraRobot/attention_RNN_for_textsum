import tensorflow as tf
import dot
import imp
imp.reload(dot)

from tensorflow.python.ops import variable_scope


sess = tf.InteractiveSession()

a=[]
t = tf.ones([1,5])
a.append(t)
for i in range(2):
   t = tf.ones([1,5]) + t
   a.append(t)


x=tf.concat(a, axis=0)

b=[]
k = tf.ones([5,4])
b.append(k)
for i in range(2):
   k=tf.ones([5,4]) + k
   b.append(k)

tmp = tf.concat(b, axis=0)
y = tf.reshape(tmp, [-1, 5, 4])

init = tf.ones_initializer()

with variable_scope.variable_scope("score") as scope:
   scores=dot.score(x, y, init)

sess.run(tf.global_variables_initializer())

tf.reset_default_graph()
