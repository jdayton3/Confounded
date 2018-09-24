"""Definitions for the neural networks in Confounded.
"""

# pylint: disable=E1129

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm # pylint: disable=E0611

class Confounded(object):
    def __init__(self, input_size, code_size, num_targets):
        self.sess = tf.Session()

        self.input_size = input_size
        self.code_size = code_size
        self.num_targets = num_targets

        self.inputs = None
        self.outputs = None
        self.targets = None
        self.classification = None
        self.optimizer = None
        self.d_optimizer = None

        self._setup_networks()

    def _setup_networks(self):
        self._setup_autoencoder()
        self._setup_discriminator()
        self._setup_loss_functions()

    def _setup_autoencoder(self):
        with tf.name_scope("autoencoder"):
            self.inputs = tf.placeholder(tf.float32, [None, self.input_size])
            self.show_image(self.inputs, "inputs")
            with tf.name_scope("encoding"):
                encode1 = fully_connected(self.inputs, 512, activation_fn=tf.nn.relu)
                encode1 = batch_norm(encode1)
                encode2 = fully_connected(encode1, 256, activation_fn=tf.nn.relu)
                encode2 = batch_norm(encode2)
                code = fully_connected(encode2, self.code_size, activation_fn=tf.nn.relu)
            with tf.name_scope("decoding"):
                decode1 = fully_connected(code, 256, activation_fn=tf.nn.relu)
                decode1 = batch_norm(decode1)
                decode2 = fully_connected(decode1, 512, activation_fn=tf.nn.relu)
                decode2 = batch_norm(decode2)
                # TODO: This shouldn't be a sigmoid because we can never make it to 0 or 1.
                # Change it to something else and squash the output to the input's [min, max]
                self.outputs = fully_connected(decode2, self.input_size, activation_fn=tf.nn.sigmoid)
            self.show_image(self.outputs, "outputs")

    def _setup_discriminator(self):
        with tf.name_scope("discriminator"):
            self.targets = tf.placeholder(tf.float32, [None, self.num_targets])
            fc1 = fully_connected(self.outputs, 256)
            fc2 = fully_connected(fc1, 128)
            fc3 = fully_connected(fc2, 64)
            fc4 = fully_connected(fc3, 8)
            fc5 = fully_connected(fc4, 8)
            # TODO: Consider changing the sigmoid + MSE to cross entropy.
            self.classification = fully_connected(fc5, self.num_targets, activation_fn=tf.nn.sigmoid)
            with tf.name_scope("optimizer"):
                d_loss = tf.losses.mean_squared_error(self.classification, self.targets)
                tf.summary.scalar("mse", d_loss)
                self.d_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(d_loss)

    def _setup_loss_functions(self):
        with tf.name_scope("discriminator"):
            with tf.name_scope("optimizer"):
                d_loss = tf.losses.mean_squared_error(self.classification, self.targets)
                tf.summary.scalar("mse", d_loss)
                self.d_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(d_loss)
        with tf.name_scope("autoencoder"):
            with tf.name_scope("optimizer"):
                mse = tf.losses.mean_squared_error(self.inputs, self.outputs)
                tf.summary.scalar("mse", mse)
                loss = mse + (tf.ones_like(d_loss) - d_loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    def show_image(self, x, name="image"):
        # This assumes the input is a square image...
        width_height = int(self.input_size**0.5)
        img = tf.reshape(x, [-1, width_height, width_height, 1])
        tf.summary.image(name, img, max_outputs=1)