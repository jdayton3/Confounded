from . import hide_warnings
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)

    inputs = tf.placeholder(tf.float32, [None, 784])
    outputs = tf.identity(inputs)
    targets = tf.placeholder(tf.float32, [None, 10])
    # TODO: make a super simple autoencoder with MNIST

    with tf.Session() as sess:
        for _ in range(1):
            batch_inputs, batch_targets = mnist.train.next_batch(100)
            _ = sess.run([outputs], feed_dict={
                inputs: batch_inputs,
                targets: batch_targets
                })