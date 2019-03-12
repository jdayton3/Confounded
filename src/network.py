"""Definitions for the neural networks in Confounded.
"""

# pylint: disable=E1129

import functools
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm # pylint: disable=E0611
from math import ceil

def is_square(n):
    sqrt = n**0.5
    return int(sqrt) == sqrt

def var_scope(scope):
    """Decorator to wrap a function in a tensorflow variable scope

    Arguments:
        scope {str} -- Name of the variable scope

    Returns:
        function -- The decorated function wrapped in the variable scope
    """
    def decorator_var_scope(func):
        @functools.wraps(func)
        def wrapper_var_scope(*args, **kwargs):
            with tf.variable_scope(scope):
                return func(*args, **kwargs)
        return wrapper_var_scope
    return decorator_var_scope

def show_before_and_after_images(func):
    @functools.wraps(func)
    def wrapper_show_images(*args, **kwargs):
        inputs = args[0]
        show_image(inputs, name="inputs")
        outputs = func(*args, **kwargs)
        if isinstance(outputs, tuple):
            # The autoencoder functions might return (outputs, loss)
            show_image(outputs[0], name="outputs")
        else:
            show_image(outputs, name="outputs")
        return outputs
    return wrapper_show_images

@var_scope("vae")
@show_before_and_after_images
def variational_autoencoder(inputs):
    """Creates a variational autoencoder based on "Hands-On Machine
    Learning with Scikit-Learn and TensorFlow by Aurélien Géron
    (O’Reilly). Copyright 2017 Aurélien Géron, 978-1-491-96229-9."

    Arguments:
        input_size {int} -- Size of the input to the autoencoder

    Returns:
        input {Tensor} -- The input tensor
        output {Tensor} -- The output tensor
        loss {Tensor} -- The loss operation
    """
    code_size = 20
    layer_sizes = [500, 500]
    activations = [tf.nn.elu for _ in layer_sizes]
    input_size = layer_size(inputs)

    encoding = make_layers(inputs, layer_sizes, activations)
    code_mean, code_gamma, code = vae_code_layer(encoding, code_size)
    decoding = make_layers(code, layer_sizes, activations)
    logits = fully_connected(decoding, input_size, activation_fn=None)
    outputs = tf.sigmoid(logits)

    reconstruction_loss = tf.losses.mean_squared_error(inputs, outputs)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits)
    reconstruction_loss = tf.reduce_sum(xentropy)
    latent_loss = kl_divergence(code_gamma, code_mean) #* 0.01 / code_size
    loss = reconstruction_loss + latent_loss

    return outputs, loss

def layer_size(layer):
    dimensions = layer.shape[1:]
    size = 1
    for dimension in dimensions:
        size *= int(dimension) # must be converted from Dimension to int
    return size

def make_layers(inputs, layer_sizes, activations=None, keep_prob=1.0, do_batch_norm=False):
    if not activations:
        activations = [tf.nn.relu for _ in layer_sizes]
    current_layer = inputs
    for layer_size, activation in zip(layer_sizes, activations):
        current_layer = fully_connected(current_layer, layer_size, activation_fn=activation)
        current_layer = tf.nn.dropout(current_layer, keep_prob)
        if do_batch_norm:
            current_layer = batch_norm(current_layer)
    return current_layer

def vae_code_layer(inputs, code_size):
    code_mean = fully_connected(inputs, code_size, activation_fn=None)
    code_gamma = fully_connected(inputs, code_size, activation_fn=None)
    noise = tf.random_normal(tf.shape(code_gamma), dtype=tf.float32)
    code = code_mean + tf.exp(0.5 * code_gamma) * noise
    return code_mean, code_gamma, code

def kl_divergence(gamma, mean):
    return 0.5 * tf.reduce_sum(tf.exp(gamma) + tf.square(mean) - 1 - gamma)

def show_image(x, name="image"):
    input_size = layer_size(x)
    if is_square(input_size):
        width_height = int(input_size**0.5)
        img = tf.reshape(x, [-1, width_height, width_height, 1])
        tf.summary.image(name, img, max_outputs=1)

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
        self.ae_loss = None
        self.loss = None
        self.optimizer = None
        self.d_optimizer = None

        self._setup_networks()

    def _setup_networks(self):
        self._setup_autoencoder()
        self._setup_discriminator()
        self._setup_loss_functions()

    @var_scope("autoencoder")
    def _setup_autoencoder(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.input_size])
        self.outputs, self.ae_loss = variational_autoencoder(self.inputs)
        # keep_prob = 0.5
        # with tf.variable_scope("autoencoder"):
            # self.inputs = tf.placeholder(tf.float32, [None, self.input_size])
            # if is_square(self.input_size):
            #     self.show_image(self.inputs, "inputs")
            # with tf.name_scope("encoding"):
            #     layer = self.inputs
            #     n_nodes = 512
            #     encoding_layers = [] # Save unactivated for u-net-like connections.
            #     for i in range(self.autoencoder_layers):
            #         layer = fully_connected(layer, n_nodes, activation_fn=None)
            #         encoding_layers.append(layer)
            #         layer = self.activation(layer)
            #         layer = batch_norm(layer)
            #         layer = tf.nn.dropout(layer, keep_prob)
            #         n_nodes = int(ceil(n_nodes / 2))
            #     self.code = fully_connected(layer, self.code_size, activation_fn=self.activation)
            #     layer = self.code
            # with tf.name_scope("decoding"):
            #     encoding_layers = reversed(encoding_layers)
            #     for i, skip_layer in enumerate(encoding_layers):
            #         n_nodes *= 2
            #         layer = fully_connected(
            #             layer, n_nodes, activation_fn=self.activation) + skip_layer
            #         layer = batch_norm(layer)
            #         layer = tf.nn.dropout(layer, keep_prob)
            #     self.outputs = fully_connected(
            #         layer, self.input_size, activation_fn=tf.nn.sigmoid)# + self.inputs
            # if is_square(self.input_size):
            #     self.show_image(self.outputs, "outputs")

    @var_scope("discriminator")
    def _setup_discriminator(self):
        keep_prob = 0.5
        self.targets = tf.placeholder(tf.float32, [None, self.num_targets])
        layer = batch_norm(self.outputs)
        layer_size = 512
        for _ in range(self.discriminator_layers):
            layer = fully_connected(layer, 512, activation_fn=self.activation)
            layer = tf.nn.dropout(layer, keep_prob)
            layer = batch_norm(layer)
            layer_size = int(ceil(layer_size / 2))
        self.logits = fully_connected(layer, self.num_targets, activation_fn=None)
        self.classification = tf.nn.sigmoid(self.logits)

    def _setup_loss_functions(self):
        with tf.name_scope("discriminator"):
            with tf.name_scope("optimizer"):
                self.d_loss = tf.losses.mean_squared_error(self.classification, self.targets)
                xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
                self.d_loss = tf.reduce_sum(xentropy)
                tf.summary.scalar("d_loss", self.d_loss)
                discriminator_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    "discriminator"
                )
                self.d_optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001
                ).minimize(self.d_loss, var_list=discriminator_vars)
        with tf.name_scope("autoencoder"):
            with tf.name_scope("optimizer"):
                # self.ae_loss = tf.losses.mean_squared_error(self.inputs, self.outputs)
                tf.summary.scalar("ae_loss", self.ae_loss)
                self.loss = self.ae_loss + (
                    tf.ones_like(self.d_loss) - self.disc_weighting * self.d_loss
                )
                tf.summary.scalar("dual_loss", self.loss)
                autoencoder_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    "autoencoder"
                )
                self.ae_optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001
                ).minimize(self.ae_loss, var_list=autoencoder_vars)
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.001
                ).minimize(self.loss, var_list=autoencoder_vars)
