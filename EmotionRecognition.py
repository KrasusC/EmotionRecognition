import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from DatasetTool import DatasetTool

# Training Parameters
learning_rate = 0.001
training_epoch = 100
batch_size = 24 # batch_size must be multiples of 3
display_step = 50
dataset_config_file_path = '/scratch/user/liqingqing/info_concatenated_with_two_SNR'
dropout_rate = 0.2

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
    # 12x6 conv, 3 input, 16 outputs
    'wconv1': tf.Variable(tf.random_normal([12, 6, 1, 16])),
    # 8x3 conv, 16 inputs, 24 outputs
    'wconv2': tf.Variable(tf.random_normal([8, 3, 16, 24])),
    # 5x2 conv, 24 inputs, 32 outputs
    'wconv3': tf.Variable(tf.random_normal([5, 2, 24, 32])),
    #attention_cnn
    #only one-dimmensional conv
    'attwconv1': tf.Variable(tf.random_normal([5, 256, 5])),
    #fc
    'attwfc1': tf.Variable(tf.random_normal([40, 37])),
    'attwfc2': tf.Variable(tf.random_normal([37, 37])),

    'cwfc1': tf.Variable(tf.random_normal([256, 128])),
    'cwfc2': tf.Variable(tf.random_normal([128, 32])),
    'cwfc3': tf.Variable(tf.random_normal([32, num_classes])),

    # 1024 inputs, 10 outputs (class prediction)
    'classify': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'bconv1': tf.Variable(tf.random_normal([16])),
    'bconv2': tf.Variable(tf.random_normal([24])),
    'bconv3': tf.Variable(tf.random_normal([32])),
    'attbconv1': tf.Variable(tf.random_normal([5])),
    'attbfc1':   tf.Variable(tf.random_normal([37])),
    'attbfc2':   tf.Variable(tf.random_normal([37])),
    'cbfc1':   tf.Variable(tf.random_normal([128])),
    'cbfc2':   tf.Variable(tf.random_normal([32])),
    'cbfc3':   tf.Variable(tf.random_normal([num_classes])),
    'classify': tf.Variable(tf.random_normal([num_classes]))
}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv1d(x, W, b, stride=5):
    x = tf.nn.conv1d(x, W, stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def convNN(x, x1):
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
    #print('conv3_shape:', conv3)
    conv3 = tf.reshape(conv3, shape=[batch_size, timesteps, -1])
    #print('conv3_shape:', conv3)
    conv3 = tf.concat([conv3, x1], 2)
    #print('concatenated_conv3_shape:', conv3)

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

    #stack as [batch_size, timesteps, 2*num_hidden]
    outputs = tf.stack(outputs, axis = 1)
    print('stacked_x_shape:', outputs)
    return outputs

#generate [batch_size, timesteps] weights
def AttentionNN(x):
    conv1 = conv1d(x, weights['attwconv1'], biases['attbconv1'])

    print('att_conv1_shape:', conv1)
    conv1 = tf.reshape(conv1, shape=[batch_size, -1])

    fc1 = tf.add(tf.matmul(conv1, weights['attwfc1']), biases['attbfc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout_rate)

    fc2 = tf.add(tf.matmul(fc1, weights['attwfc2']), biases['attbfc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout_rate)

    return fc2

def ClassifyNN(x, a):
    #add the [1] dimmension for broadcasting
    a = tf.reshape(a, shape=[batch_size, timesteps, 1])
    x = tf.multiply(x, a)

    print('class_x_shape:', x)
    x = tf.reduce_mean(x, 1)
    print('class_x_mean_shape:', x)

    fc1 = tf.add(tf.matmul(x, weights['cwfc1']), biases['cbfc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout_rate)

    fc2 = tf.add(tf.matmul(fc1, weights['cwfc2']), biases['cbfc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout_rate)

    fc3 = tf.add(tf.matmul(fc2, weights['cwfc3']), biases['cbfc3'])

    return fc3

#Full NN architecture
outputs = convNN(X, X_1)
lstm_output = BiRNN(outputs)
attention_para = AttentionNN(lstm_output)
logits = ClassifyNN(lstm_output, attention_para)
#prediction = tf.nn.softmax(logits)
prediction = tf.div(logits, tf.reshape(tf.reduce_sum(logits, 1), shape=[batch_size, 1]))

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
        train_stat_cnt = 0

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
                train_stat_cnt += 1

        print("Training Epoch " + str(epoch_num) + ", Epoch Loss= " + \
              "{:.4f}".format(total_train_loss / train_stat_cnt) + ", Epoch Training Accuracy= " + \
              "{:.3f}".format(total_train_accuracy / train_stat_cnt))


        if epoch_num % 5 == 0: #test every 5 epoch
            for step in range(1, testing_steps + 1):
                batch_x, batch_x_1, batch_y = dataset.next_batch(is_train = False)
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, X_1: batch_x_1, Y: batch_y})

                total_test_loss += loss
                total_test_accuracy += acc

                if step % display_step == 0 or step == 1:
                    print("Testing Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Testing Accuracy= " + \
                          "{:.3f}".format(acc))


            print("Test of Training Epoch " + str(epoch_num) + ", Total Testing Loss= " + \
                  "{:.4f}".format(total_test_loss / testing_steps) + ", Epoch Testing Accuracy= " + \
                  "{:.3f}".format(total_test_accuracy / testing_steps))
