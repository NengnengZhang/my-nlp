import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import com.hello.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer_1'):
    with tf.name_scope('Weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='Weights')
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='biases')
    with tf.name_scope('y'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope('total'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

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
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 50 == 0:
        summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(summary, i)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print('Accuracy at step %s: %s' % (i, acc))

# 测试一下模型
test_xs, test_ys = mnist.train.next_batch(1)
y_pre = sess.run(y, feed_dict={x: test_xs, y_: mnist.test.labels})

print(y_pre)
print(np.argmax(y_pre))

# 显示图像
plt.figure()

im = test_xs[0].reshape(28, 28)
plt.imshow(im, 'gray')
plt.pause(0.0000001)
plt.show()
