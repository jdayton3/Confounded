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
                 autoencoder_layers=2,
                 activation=tf.nn.relu,
                 disc_weghting=1.0):
        self.sess = tf.Session()

        self.input_size = input_size
        self.code_size = code_size
        self.num_targets = num_targets
        self.discriminator_layers = discriminator_layers
        self.autoencoder_layers = autoencoder_layers
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
        keep_prob = 0.5
        with tf.variable_scope("autoencoder"):
            self.inputs = tf.placeholder(tf.float32, [None, self.input_size])
            if is_square_image:
                self.show_image(self.inputs, "inputs")
            with tf.name_scope("encoding"):
                layer = self.inputs
                n_nodes = 512
                encoding_layers = [] # Save unactivated for u-net-like connections.
                for i in range(self.autoencoder_layers):
                    layer = fully_connected(layer, n_nodes, activation_fn=None)
                    encoding_layers.append(layer)
                    layer = self.activation(layer)
                    layer = batch_norm(layer)
                    layer = tf.nn.dropout(layer, keep_prob)
                    n_nodes = int(ceil(n_nodes / 2))
                self.code = fully_connected(layer, self.code_size, activation_fn=self.activation)
                layer = self.code
            with tf.name_scope("decoding"):
                encoding_layers = reversed(encoding_layers)
                for i, skip_layer in enumerate(encoding_layers):
                    n_nodes *= 2
                    layer = fully_connected(
                        layer, n_nodes, activation_fn=self.activation) + skip_layer
                    layer = batch_norm(layer)
                    layer = tf.nn.dropout(layer, keep_prob)
                self.outputs = fully_connected(
                    layer, self.input_size, activation_fn=tf.nn.sigmoid)# + self.inputs
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
