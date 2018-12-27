"""Definitions for the neural networks in Confounded.
"""

# pylint: disable=E1129

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm # pylint: disable=E0611
from math import ceil

class Confounded(object):
    def __init__(self,
                 input_size,
                 code_size,
                 num_targets,
                 discriminator_layers=2,
                 activation=tf.nn.relu,
                 disc_weghting=1.0):
        self.sess = tf.Session()

        self.input_size = input_size
        self.code_size = code_size
        self.num_targets = num_targets
        self.discriminator_layers = discriminator_layers
        self.activation = activation
        self.disc_weighting = disc_weghting

        self.inputs = None
        self.code = None
        self.outputs = None
        self.targets = None
        self.classification = None

        self.d_loss = None
        self.mse = None
        self.loss = None
        self.optimizer = None
        self.d_optimizer = None

        self._setup_networks()

    def _setup_networks(self):
        self._setup_autoencoder()
        self._setup_discriminator()
        self._setup_loss_functions()

    def _setup_autoencoder(self):
        sqrt = self.input_size ** 0.5
        is_square_image = sqrt == int(sqrt)
        with tf.variable_scope("autoencoder"):
            self.inputs = tf.placeholder(tf.float32, [None, self.input_size])
            if is_square_image:
                self.show_image(self.inputs, "inputs")
            with tf.name_scope("encoding"):
                encode1 = fully_connected(self.inputs, 512, activation_fn=self.activation)
                encode1 = batch_norm(encode1)
                encode2 = fully_connected(encode1, 256, activation_fn=self.activation)
                encode2 = batch_norm(encode2)
                self.code = fully_connected(encode2, self.code_size, activation_fn=self.activation)
            with tf.name_scope("decoding"):
                decode1 = fully_connected(self.code, 256, activation_fn=self.activation)
                decode1 = batch_norm(decode1)
                decode2 = fully_connected(decode1, 512, activation_fn=self.activation)
                decode2 = batch_norm(decode2)
                self.outputs = fully_connected(decode2, self.input_size, activation_fn=tf.nn.sigmoid)
            if is_square_image:
                self.show_image(self.outputs, "outputs")

    def _setup_discriminator(self):
        keep_prob = 0.5
        with tf.variable_scope("discriminator"):
            self.targets = tf.placeholder(tf.float32, [None, self.num_targets])
            layer = batch_norm(self.outputs)
            layer_size = 512
            for _ in range(self.discriminator_layers):
                layer = fully_connected(layer, 512, activation_fn=self.activation)
                layer = tf.nn.dropout(layer, keep_prob)
                layer = batch_norm(layer)
                layer_size = int(ceil(layer_size / 2))
            self.classification = fully_connected(layer, self.num_targets, activation_fn=tf.nn.sigmoid)

    def _setup_loss_functions(self):
        with tf.name_scope("discriminator"):
            with tf.name_scope("optimizer"):
                self.d_loss = tf.losses.mean_squared_error(self.classification, self.targets)
                tf.summary.scalar("mse", self.d_loss)
                discriminator_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    "discriminator"
                )
                self.d_optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001
                ).minimize(self.d_loss, var_list=discriminator_vars)
        with tf.name_scope("autoencoder"):
            with tf.name_scope("optimizer"):
                self.mse = tf.losses.mean_squared_error(self.inputs, self.outputs)
                tf.summary.scalar("mse", self.mse)
                self.loss = self.mse + (
                    tf.ones_like(self.d_loss) - self.disc_weighting * self.d_loss
                )
                tf.summary.scalar("dual_loss", self.loss)
                autoencoder_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    "autoencoder"
                )
                self.ae_optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001
                ).minimize(self.mse, var_list=autoencoder_vars)
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001
                ).minimize(self.loss, var_list=autoencoder_vars)

    def show_image(self, x, name="image"):
        # This assumes the input is a square image...
        width_height = int(self.input_size**0.5)
        img = tf.reshape(x, [-1, width_height, width_height, 1])
        tf.summary.image(name, img, max_outputs=1)
