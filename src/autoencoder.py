# pylint: disable=E1129,E0611

from . import hide_warnings
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

INPUT_SIZE = 784
CODE_SIZE = 200
BATCH_SIZE = 100
NUM_TARGETS = 2

# Statistics the normal distribution should have
MEAN = 0.0
STDEV = 1.0


def show_image(x, name="image"):
    # This assumes the input is a square image...
    width_height = int(INPUT_SIZE**0.5)
    img = tf.reshape(x, [-1, width_height, width_height, 1])
    tf.summary.image(name, img, max_outputs=1)

def mask(inputs):
    return 1.0 - inputs

if __name__ == "__main__":
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)

    # Autoencoder net
    with tf.name_scope("autoencoder"):
        inputs = tf.placeholder(tf.float32, [None, INPUT_SIZE])
        show_image(inputs, "inputs")
        with tf.name_scope("encoding"):
            encode1 = fully_connected(inputs, 512, activation_fn=tf.nn.relu)
            encode2 = fully_connected(encode1, 256, activation_fn=tf.nn.relu)
            code = fully_connected(encode2, CODE_SIZE, activation_fn=tf.nn.relu)
        with tf.name_scope("decoding"):
            decode1 = fully_connected(code, 256, activation_fn=tf.nn.relu)
            decode2 = fully_connected(decode1, 512, activation_fn=tf.nn.relu)
            outputs = fully_connected(decode2, INPUT_SIZE, activation_fn=tf.nn.sigmoid)
        show_image(outputs, "outputs")

    # Discriminator net
    with tf.name_scope("discriminator"):
        targets = tf.placeholder(tf.float32, [None, NUM_TARGETS])
        fc1 = fully_connected(outputs, 256)
        fc2 = fully_connected(fc1, 128)
        fc3 = fully_connected(fc2, 64)
        fc4 = fully_connected(fc3, 8)
        fc5 = fully_connected(fc4, 8)
        classification = fully_connected(fc5, NUM_TARGETS, activation_fn=tf.nn.sigmoid)
        with tf.name_scope("optimizer"):
            d_loss = tf.losses.mean_squared_error(classification, targets)
            tf.summary.scalar("mse", d_loss)
            d_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(d_loss)

    with tf.name_scope("autoencoder"):
        with tf.name_scope("optimizer"):
            mse = tf.losses.mean_squared_error(inputs, outputs)
            tf.summary.scalar("mse", mse)
            loss =  mse + (tf.ones_like(d_loss) - d_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/adversarial", sess.graph)
        tf.global_variables_initializer().run()

        for i in range(10000):
            batch_inputs, _ = mnist.train.next_batch(BATCH_SIZE)
            if i % 2 == 0:
                batch_inputs = mask(batch_inputs)
                target = [[1.0, 0.0]] * BATCH_SIZE
            else:
                target = [[0.0, 1.0]] * BATCH_SIZE
            target = np.array(target)
            summary, disc, _, _ = sess.run([merged, outputs, optimizer, d_optimizer], feed_dict={
                inputs: batch_inputs,
                targets: target,
            })
            writer.add_summary(summary, i)

            #TODO: change the masking layer, trim/normalize the weights