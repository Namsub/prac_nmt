from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.python.client import timeline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
training_steps = 1
batch_size = 128
display_step = 200

# Network parameters
num_input = 28
timesteps = 28
num_hidden = 512
num_classes = 10

# PATH
PATH = "/home/titanxp/prac_tensor"

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
		# Hidden layer weights => 2*n_hidden because of forward + backward cells
		'out' : tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
		'out' : tf.Variable(tf.random_normal([num_classes]))
}

def BiRNN(x, weights, biases):
	# Prepare data shape to match 'rnn' function requirements
	# Current data input shape: (batch_size, timesteps, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
	x = tf.unstack(x, timesteps, 1)

	# Define lstm cells with tensorflow
	# Forward direction cell
	lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	# Backward direction cell
	lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

	# Get lstm cell output
	try:
		outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

	except Exception:	# Old Tensorflow version only returns outputs not states
		outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']



logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
# Start training
with tf.Session() as sess:
	# Run the initializer
	sess.run(init)

	for step in range(1, training_steps+1):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		# Reshape data to get 28 seq of 28 elements
		batch_x = batch_x.reshape((batch_size, timesteps, num_input))
		# Run optimization op (backprop)
		sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
		#sess.run(train_op, feed_dict={X: batch_x, Y: batch_y}, options=run_options, run_metadata=run_metadata)
		if step % display_step == 0 or step == 1:
			# Calculate batch loss and accuracy
			#loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y}, options=run_options, run_metadata=run_metadata)
			tl = timeline.Timeline(step_stats=run_metadata.step_stats)
			ctf = tl.generate_chrome_trace_format(show_memory=True)
			print("Step")

		trace_name = "trace_" + str(step) + ".json"
		trace_file = PATH + '/' + trace_name
		with open(trace_file, 'w') as f:
			f.write(ctf)
	
	print("Optimization Finished!")

	#'''
	# Calculate accuracy for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={X:test_data, Y:test_label}))
	#'''
