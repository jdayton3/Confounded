# pylint: disable=E1129,E0611,E1101

from . import hide_warnings
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
import pandas as pd
from . import reformat
from .load_data import split_features_labels

INPUT_PATH = "./data/tidy_batches_balanced.csv"
OUTPUT_PATH = "./data/tidy_confounded_balanced.csv"
META_COLS = None
MINIBATCH_SIZE = 100
CODE_SIZE = 200

def show_image(x, name="image"):
    # This assumes the input is a square image...
    width_height = int(INPUT_SIZE**0.5)
    img = tf.reshape(x, [-1, width_height, width_height, 1])
    tf.summary.image(name, img, max_outputs=1)

def categorical_columns(df):
    """Get the names of all categorical columns in the dataframe.

    Arguments:
        df {pandas.DataFrame} -- The dataframe.

    Returns:
        list -- Names of the categorical columns in the dataframe.
    """
    return list(df.select_dtypes(exclude=['float']).columns)

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

        self.setup_networks()

    def setup_networks(self):
        self.autoencoder()
        self.discriminator()
        self.loss_functions()

    def autoencoder(self):
        with tf.name_scope("autoencoder"):
            self.inputs = tf.placeholder(tf.float32, [None, self.input_size])
            show_image(self.inputs, "inputs")
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
                self.outputs = fully_connected(decode2, self.input_size, activation_fn=tf.nn.sigmoid)
            show_image(self.outputs, "outputs")

    def discriminator(self):
        with tf.name_scope("discriminator"):
            self.targets = tf.placeholder(tf.float32, [None, self.num_targets])
            fc1 = fully_connected(self.outputs, 256)
            fc2 = fully_connected(fc1, 128)
            fc3 = fully_connected(fc2, 64)
            fc4 = fully_connected(fc3, 8)
            fc5 = fully_connected(fc4, 8)
            self.classification = fully_connected(fc5, self.num_targets, activation_fn=tf.nn.sigmoid)
            with tf.name_scope("optimizer"):
                d_loss = tf.losses.mean_squared_error(self.classification, self.targets)
                tf.summary.scalar("mse", d_loss)
                self.d_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(d_loss)

    def loss_functions(self):
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

if __name__ == "__main__":
    # Get sizes & meta cols
    data = pd.read_csv(INPUT_PATH)
    if META_COLS is None:
        META_COLS = categorical_columns(data)
        print "Inferred meta columns:", META_COLS
    INPUT_SIZE = len(data.columns) - len(META_COLS)
    NUM_TARGETS = len(data["Batch"].unique())

    # TODO: encapsulate the networks in a class or function.
    # If it's a class, it could have a "train" method & would need to take targets and inputs as parameters.
    # Other possible methods might include training only the autoencoder or only the discriminator.
    # Other possible parameters might include the depth of the network, learning rate, minibatch size, & the encoding layer size.
    # Autoencoder net
    c = Confounded(INPUT_SIZE, CODE_SIZE, NUM_TARGETS)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/ae_class", sess.graph)
        tf.global_variables_initializer().run()

        # Train
        for i in range(100):
            features, labels = split_features_labels(
                data.sample(MINIBATCH_SIZE, replace=True),
                meta_cols=META_COLS
            )
            summary, disc, out, _ = sess.run([merged, c.outputs, c.optimizer, c.d_optimizer], feed_dict={
                c.inputs: features,
                c.targets: labels,
            })
            writer.add_summary(summary, i)

        # Run the csv through confounded
        features, labels = split_features_labels(data, meta_cols=META_COLS)
        adj, = sess.run([c.outputs], feed_dict={
            c.inputs: features,
            c.targets: labels,
        })
        # Save adjusted & non-adjusted numbers
        df_adj = pd.DataFrame(adj, columns=list(range(INPUT_SIZE)))
        reformat.to_csv(
            df_adj,
            OUTPUT_PATH,
            tidy=True,
            meta_cols={
                col: data[col] for col in META_COLS
            }
        )
