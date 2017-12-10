from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from DatasetTool import DatasetTool

# Training Parameters
learning_rate = 0.001
training_epoch = 100
batch_size = 30 # batch_size must be multiples of 3
display_step = 5
dataset_config_file_path = '/scratch/user/liqingqing/info_concatenated'
dropout_rate = 0.3

# Network Parameters
input_x = 296
input_y = 26
input_z = 1
input_1 = 4 # extra vector size example
timesteps = 37 # timesteps = 296 / 8
num_hidden = 128 # hidden layer num of features
num_classes = 6 # total classes (0-9 digits)

dataset = DatasetTool(dataset_config_file_path, batch_size, timesteps)
training_steps, testing_steps = dataset.get_steps()

# tf Graph input
X = tf.placeholder("float", [batch_size, input_x, input_y])
X_1 = tf.placeholder("float", [batch_size, timesteps, input_1])
Y = tf.placeholder("float", [batch_size, num_classes])

# Define weights
weights = {
    # 12x16 conv, 3 input, 16 outputs
    'wconv1': tf.Variable(tf.random_normal([12, 6, 1, 16])),
    # 8x12 conv, 16 inputs, 24 outputs
    'wconv2': tf.Variable(tf.random_normal([8, 3, 16, 24])),
    # 5x7 conv, 24 inputs, 32 outputs
    'wconv3': tf.Variable(tf.random_normal([5, 2, 24, 32])),
    # 1024 inputs, 10 outputs (class prediction)
    'classify': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'bconv1': tf.Variable(tf.random_normal([16])),
    'bconv2': tf.Variable(tf.random_normal([24])),
    'bconv3': tf.Variable(tf.random_normal([32])),
    'classify': tf.Variable(tf.random_normal([num_classes]))
}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def convNet(x, x1):
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, input_x, input_y, input_z])
    # Convolution Layer
    #print('x_shape:', x)
    print('x1_shape:', x1)
    conv1 = conv2d(x, weights['wconv1'], biases['bconv1'])
    conv1 = maxpool2d(conv1, k=2)

    #print('conv1_shape:', tf.shape(conv1))

    conv2 = conv2d(conv1, weights['wconv2'], biases['bconv2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wconv3'], biases['bconv3'])
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    print('conv3_shape:', conv3)
    conv3 = tf.reshape(conv3, shape=[batch_size, timesteps, -1])
    print('conv3_shape:', conv3)
    conv3 = tf.concat([conv3, x1], 2)
    print('concatenated_conv3_shape:', conv3)

    #fc1 = tf.add(tf.matmul(conv3, weights['wfc1']), biases['bfc1'])
    #fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout_rate)

    return conv3


def BiRNN(x):
    x = tf.unstack(x, timesteps, 1)
    #print('unpacked_lstm_input_shape:', x)
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    logits =  tf.matmul(outputs[-1], weights['classify']) + biases['classify']
    return logits

#Full NN architecture
outputs = convNet(X, X_1)

logits = BiRNN(outputs)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
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

    print('Exp Total Epoch:', training_epoch)
    print('Exp Batch Size:', batch_size)
    print('Exp Training Steps:', training_steps)
    print('Exp Testing Steps:', testing_steps)
    for epoch_num in range(1, training_epoch + 1):
        total_train_accuracy = 0
        total_test_accuracy = 0
        total_train_loss = 0
        total_test_loss = 0

        for step in range(1, training_steps + 1):
            batch_x, batch_x_1, batch_y = dataset.next_batch(is_train = True)
            sess.run(train_op, feed_dict={X: batch_x, X_1: batch_x_1, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, X_1: batch_x_1, Y: batch_y})
                print("Training Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                total_train_loss += loss
                total_train_accuracy += acc

        print("Training Epoch " + str(epoch_num) + ", Epoch Loss= " + \
              "{:.4f}".format(total_train_loss / training_steps) + ", Epoch Training Accuracy= " + \
              "{:.3f}".format(total_train_accuracy / training_steps))


        if epoch_num % 5 == 0: #test every 5 epoch
            for step in range(1, testing_steps + 1):
                batch_x, batch_x_1, batch_y = dataset.next_batch(is_train = False)
                sess.run(accuracy, feed_dict={X: batch_x, X_1: batch_x_1, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Testing Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Testing Accuracy= " + \
                          "{:.3f}".format(acc))
                    total_test_loss += loss
                    total_test_accuracy += acc

            print("Training Etep " + str(epoch_num) + ", Total Testing Loss= " + \
                  "{:.4f}".format(total_test_loss / testing_steps) + ", Epoch Testing Accuracy= " + \
                  "{:.3f}".format(total_test_accuracy / testing_steps))
