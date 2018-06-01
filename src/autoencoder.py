# pylint: disable=E1129,E0611,E1101

from . import hide_warnings
from .adjustments import Noise
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

INPUT_SIZE = 784
CODE_SIZE = 200
BATCH_SIZE = 100
NUM_TARGETS = 2
NOISE = np.random.normal(size=INPUT_SIZE)

# Statistics the normal distribution should have
MEAN = 0.0
STDEV = 1.0


def show_image(x, name="image"):
    # This assumes the input is a square image...
    width_height = int(INPUT_SIZE**0.5)
    img = tf.reshape(x, [-1, width_height, width_height, 1])
    tf.summary.image(name, img, max_outputs=1)

def mult_noise(inputs):
    outs = inputs * NOISE
    outs *= 1.0 / outs.max()
    return outs

def add_noise(inputs):
    outs = inputs + 0.1 * NOISE
    outs *= 1.0 / outs.max()
    return outs

def invert(inputs):
    return 1.0 - inputs

if __name__ == "__main__":
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)

    # Autoencoder net
    with tf.name_scope("autoencoder"):
        inputs = tf.placeholder(tf.float32, [None, INPUT_SIZE])
        show_image(inputs, "inputs")
        with tf.name_scope("encoding"):
            encode1 = fully_connected(inputs, 512, activation_fn=tf.nn.relu)
            encode1 = batch_norm(encode1)
            encode2 = fully_connected(encode1, 256, activation_fn=tf.nn.relu)
            encode2 = batch_norm(encode2)
            code = fully_connected(encode2, CODE_SIZE, activation_fn=tf.nn.relu)
        with tf.name_scope("decoding"):
            decode1 = fully_connected(code, 256, activation_fn=tf.nn.relu)
            decode1 = batch_norm(decode1)
            decode2 = fully_connected(decode1, 512, activation_fn=tf.nn.relu)
            decode2 = batch_norm(decode2)
            outputs = fully_connected(decode2, INPUT_SIZE, activation_fn=tf.nn.sigmoid)
        show_image(outputs, "outputs")

    # Discriminator net
    with tf.name_scope("discriminator"):
        targets = tf.placeholder(tf.float32, [None, NUM_TARGETS])
        # fc1 = fully_connected(outputs, 256)
        fc1 = fully_connected(inputs, 256)
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
        writer = tf.summary.FileWriter("log/order_2_discriminator_0.001", sess.graph)
        tf.global_variables_initializer().run()
        mask = add_noise

        df = 0.001
        noiser_0 = Noise((INPUT_SIZE,), discount_factor=df)
        noiser_1 = Noise((INPUT_SIZE,), discount_factor=df)

        for i in range(10000):
            batch_inputs, _ = mnist.train.next_batch(BATCH_SIZE)
            if i % 2 == 0:
                # batch_inputs = mask(batch_inputs)
                adj_batch_inputs = []
                for x in batch_inputs:
                    adj_batch_inputs.append(noiser_0.adjust(x))
                target = [[1.0, 0.0]] * BATCH_SIZE
            else:
                adj_batch_inputs = []
                for x in batch_inputs:
                    adj_batch_inputs.append(noiser_1.adjust(x))
                target = [[0.0, 1.0]] * BATCH_SIZE
            batch_inputs = adj_batch_inputs
            target = np.array(target)
            # summary, disc, _, _ = sess.run([merged, outputs, optimizer, d_optimizer], feed_dict={
            summary, disc, _ = sess.run([merged, outputs, d_optimizer], feed_dict={
                inputs: batch_inputs,
                targets: target,
            })
            writer.add_summary(summary, i)

            #TODO: make some linear mask, test the AE vs ComBat (and eventually SVA), repeat w/ nonlinear mask
            #TODO: get committee meeting (hopefully) for 2nd week of April & send out prospectus ~1wk prior, schedule for TMCB
