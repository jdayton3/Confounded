# pylint: disable=E1129,E0611,E1101

from . import hide_warnings
from .adjustments import Noise
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
import numpy as np
import pandas as pd
from . import reformat

INPUT_PATH = "./data/tidy_batches.csv"
INPUT_SIZE = 784
CODE_SIZE = 200
BATCH_SIZE = 100
NUM_TARGETS = 2

def show_image(x, name="image"):
    # This assumes the input is a square image...
    width_height = int(INPUT_SIZE**0.5)
    img = tf.reshape(x, [-1, width_height, width_height, 1])
    tf.summary.image(name, img, max_outputs=1)

if __name__ == "__main__":
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
        writer = tf.summary.FileWriter("log/ae_csv", sess.graph)
        tf.global_variables_initializer().run()

        data = pd.read_csv(INPUT_PATH)
        meta_cols = ["Sample", "Batch"]

        # Train
        for i in range(10000):
            minibatch = data.sample(BATCH_SIZE, replace=True)
            labels = np.array(pd.get_dummies(minibatch["Batch"]), dtype=float)
            features = np.array(minibatch.drop(meta_cols, axis=1))
            summary, disc, out, _ = sess.run([merged, outputs, optimizer, d_optimizer], feed_dict={
                inputs: features,
                targets: labels,
            })
            writer.add_summary(summary, i)

        # Run the csv through confounded
        labels = np.array(pd.get_dummies(data["Batch"]), dtype=float)
        features = np.array(data.drop(meta_cols, axis=1))
        adj, = sess.run([outputs], feed_dict={
            inputs: features,
            targets: labels,
        })
        # Save adjusted & non-adjusted numbers
        df_adj = pd.DataFrame(
            adj,
            columns=list(range(INPUT_SIZE)),
            batch=data["Batch"],
            sample=data["Sample"]
        )
        reformat.to_csv(df_adj, "./data/tidy_confounded2.csv", tidy=True)
