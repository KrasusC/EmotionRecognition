from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

import DatasetTool

# Training Parameters
learning_rate = 0.001
training_epoch = 100
training_steps = 10000
testing_steps = 100
batch_size = 16
display_step = 200
dataset_config_file_path = ''
dropout_rate = 0.3

dataset = DatasetTool(dataset_config_file_path, batch_size)

# Network Parameters
input_x = 400
input_y = 300
input_z = 3
input_1 = 20 # extra vector size example
timesteps = 5 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 6 # total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [batch_size, timesteps, input_x, input_y, input_z])
X_1 = tf.placeholder("float", [batch_size, timesteps, input_1])
Y = tf.placeholder("float", [batch_size, num_classes])

# Define weights
weights = {
    # 12x16 conv, 3 input, 16 outputs
    'wconv1': tf.Variable(tf.random_normal([12, 16, 3, 16])),
    # 8x12 conv, 16 inputs, 24 outputs
    'wconv2': tf.Variable(tf.random_normal([8, 12, 16, 24])),
    # 5x7 conv, 24 inputs, 32 outputs
    'wconv2': tf.Variable(tf.random_normal([5, 7, 24, 32])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wfc1': tf.Variable(tf.random_normal([100, 128])),#TODO vector size tbd, the number should be conv3 + x1
    # 1024 inputs, 10 outputs (class prediction)
    'classify': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'bconv1': tf.Variable(tf.random_normal([16])),
    'bconv2': tf.Variable(tf.random_normal([24])),
    'bconv3': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([128])),
    'classify': tf.Variable(tf.random_normal([num_classes]))
}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def convNet(x, x1):
    # Convolution Layer
    conv1 = conv2d(x, weights['wconv1'], biases['bconv1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wconv2'], biases['bconv2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wconv3'], biases['bconv3'])
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    conv3 = tf.reshape(conv3, [batch_size*timesteps, -1])
    conv3 = tf.concat([conv3, x1], 1)
    #fc1 = tf.reshape(conv3, [-1, weights['wfc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(conv3, weights['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout_rate)

    return fc1


def BiRNN(x):
    x = tf.unstack(x, timesteps, 1)
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    return outputs

def classifyNet(inputs):
    # Linear activation, using rnn inner loop last output
    logits =  tf.matmul(inputs[-1], weights['classify']) + biases['classify']
    return tf.nn.softmax(logits)


#Full NN architecture
X_reshape = tf.reshape(X, [batch_size * timesteps, input_x, input_y, input_z]) #unpack timesteps*batch_size
X_1_reshape = tf.reshape(X_1, [batch_size * timesteps, input_1])

outputs = convNet(X_reshape, X_1_reshape)

logits = BiRNN(outputs)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for epoch_num in range(1, training_epoch + 1):
        for step in range(1, training_steps + 1):
            batch_x, batch_y = dataset.next_train_batch()#TODO batch_x should be diveided into x and x_1
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Training Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        if epoch_num % 5 == 0: #test every 5 epoch
            for step in range(1, testing_steps + 1):
                batch_x, batch_y = dataset.next_test_batch()
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Testing Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
