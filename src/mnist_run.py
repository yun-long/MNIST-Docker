import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import utility as u
import constant as C
# load data
mnist = input_data.read_data_sets(train_dir='../data',one_hot='True')

# print data information
print("Training Data Size:{0}".format(len(mnist.train.images)))
print("Test Data Size:{0}".format(len(mnist.test.images)))
print("Validation Data Size:{0}".format(len(mnist.validation.images)))

# plot a data sample
images = mnist.test.images[0:9]
labels = mnist.test.labels[0:9]
#u.plot_images(images, labels)

# input tensors
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=(None, C.img_flatten))
    y_true = tf.placeholder(tf.float32, shape=(None, C.num_classes))

# weights and biases
with tf.name_scope('weights'):
    w = tf.Variable(tf.random_normal([C.img_flatten, C.num_classes]))
with tf.name_scope('biases'):
    b = tf.Variable(tf.random_normal([C.num_classes]))

with tf.name_scope('softmax'):
    logits = tf.matmul(x,w) + b
    y_pred = tf.nn.softmax(logits)

with tf.name_scope('cost'):
    # cost function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

with tf.name_scope('optimizer'):
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

with tf.name_scope('Accuracy'):
    # performance measures
    correct_predict = tf.equal(tf.argmax(y_true,1), tf.argmax(y_pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

# create a summary into a single "operation" which we can excute in a session
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)

summary_op = tf.summary.merge_all()

episode = int(len(mnist.train.images) / C.batch_size)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logdir='../log', graph=tf.get_default_graph())
    feed_test = {x: mnist.test.images, y_true: mnist.test.labels}
    for step in range(episode):
        batch_x, batch_y = mnist.train.next_batch(C.batch_size)
        feed_train = {x: batch_x, y_true: batch_y}
        _, summary = session.run([optimizer, summary_op], feed_dict=feed_train)
        summary_writer.add_summary(summary, step * C.batch_size )
        if step % 10 == 0:
            print("Accuracy: {0}".format(accuracy.eval(feed_dict = feed_test)))




