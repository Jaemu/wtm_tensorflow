# Import tensorflow and other libraries.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import math

#matplotlib inline
import matplotlib.pyplot as pylab


sess = None
def resetSession():
	tf.reset_default_graph()
	global sess
	if sess is not None: sess.close()
	sess = tf.InteractiveSession()

resetSession()

# Create input data using NumPy. y = x * 0.1 + 0.3 + noise
x_train = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.01, size=len(x_train))
y_train = x_train * 0.1 + 0.3 + noise

# Uncomment the following line to plot our input data.
pylab.plot(x_train, y_train, '.')


# Create some fake evaluation data
x_eval = np.random.rand(len(x_train)).astype(np.float32)
noise = np.random.normal(scale=0.01, size=len(x_train))
y_eval = x_eval * 0.1 + 0.3 + noise

# Build inference graph.
# Create Variables W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_normal([1]), name='weights')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Uncomment the following lines to see what W and b are.
print(str(W))
print(str(b))

# Create a placeholder we'll use later to feed x's into the graph for training and eval.
# shape=[None] means we can put in any number of examples. 
# This is used for minibatch training, and to evaluate a lot of examples at once.
# x is the internal name tensorflow is going to use to represent our data set
x = tf.placeholder(shape=[None], dtype=tf.float32, name='x')

# Uncomment this line to see what x is
print(x)

# This is the same as tf.add(tf.mul(W, x), b), but looks nicer
y = W * x + b

sw = tf.train.SummaryWriter('summaries/', graph=tf.get_default_graph())

# Create a placeholder we'll use later to feed the correct y value into the graph
y_label = tf.placeholder(shape=[None], dtype=tf.float32, name='y_label')
print ("Y label:" + str(y_label))

# Build training graph.
loss = tf.reduce_mean(tf.square(y - y_label))  # Create an operation that calculates loss.
optimizer = tf.train.GradientDescentOptimizer(0.5)  # Create an optimizer.
train = optimizer.minimize(loss)  # Create an operation that minimizes loss.

# Uncomment the following 3 lines to see what 'loss', 'optimizer' and 'train' are.
print("loss:", loss)
print("optimizer:", optimizer)
print("train:", train)

# Create an operation to initialize all the variables.
init = tf.initialize_all_variables()
print(init)
sess.run(init)

print(sess.run([W, b]))

# Uncomment these lines to test that we can compute a y from an x (without having trained anything). 
# x must be a vector, hence [3] not just 3.
x_in = [3]
sess.run(y, feed_dict={x: x_in})

# Calculate loss on the evaluation data before training
def eval_loss():
	return sess.run(loss, feed_dict={x: x_eval, y_label: y_eval})

eval_loss()

# Track of how loss changes, so we can visualize it in TensorBoard
tf.scalar_summary('loss', loss)
summary_op = tf.merge_all_summaries()

# Perform training.
for step in range(201):
	# Run the training op; feed the training data into the graph
	summary_str, _ = sess.run([summary_op, train], feed_dict={x: x_train, y_label: y_train})
	sw.add_summary(summary_str, step)
	# Uncomment the following two lines to watch training happen real time.
	if step % 20 == 0:
		print(step, sess.run([W, b]))

# Uncomment the following lines to plot the predicted values
pylab.plot(x_train, y_train, '.', label="target")
pylab.plot(x_train, sess.run(y, feed_dict={x: x_train, y_label: y_train}), label="predicted")
pylab.legend()
pylab.show()

# Check accuracy on eval data after training
eval_loss()