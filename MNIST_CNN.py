# Convolutional Neural Network，CNN

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import com.hello.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def convolutional2d(x, Weight):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, Weight, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x-input')  # 28x28
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # change xs for the CNN
    x_image = tf.reshape(xs, [-1, 28, 28, 1])

# convolutional layer 1
with tf.name_scope('convolutional_layer_1'):
    W_convolutional_1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
    b_convolutional_1 = bias_variable([32])
    h_convolutional_1 = tf.nn.relu(convolutional2d(x_image, W_convolutional_1) + b_convolutional_1)  # output size 28x28x32
    h_pool_1 = max_pooling_2x2(h_convolutional_1)  # output size 14x14x32

# convolutional layer 2
with tf.name_scope('convolutional_layer_2'):
    W_convolutional_2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
    b_convolutional_2 = bias_variable([64])
    h_convolutional_2 = tf.nn.relu(convolutional2d(h_pool_1, W_convolutional_2) + b_convolutional_2)  # output size 14x14x64
    h_pool_2 = max_pooling_2x2(h_convolutional_2)  # output size 7x7x64

# full connection layer 1
with tf.name_scope('full_connection_layer_1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool_2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# full connection layer 2
with tf.name_scope('full_connection_layer_2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('total'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, y_: batch_ys, keep_prob: 0.5})

    if i % 50 == 0:
        summary = sess.run(merged, feed_dict={xs: batch_xs, y_: batch_ys, keep_prob: 0.5})
        writer.add_summary(summary, i)
        acc = sess.run(accuracy, feed_dict={xs: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5})
        print('Accuracy at step %s: %s' % (i, acc))

# 测试一下模型
test_xs, test_ys = mnist.train.next_batch(1)
y_pre = sess.run(y, feed_dict={xs: test_xs, y_: mnist.test.labels})

print(y_pre)
print(np.argmax(y_pre))

# 显示图像
plt.figure()

im = test_xs[0].reshape(28, 28)
plt.imshow(im, 'gray')
plt.pause(0.0000001)
plt.show()
