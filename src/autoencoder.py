# pylint: disable=E1129,E0611,E1101

from . import hide_warnings
import tensorflow as tf
import pandas as pd
from . import reformat
from .load_data import split_features_labels, list_categorical_columns
from .network import Confounded

INPUT_PATH = "./data/tidy_batches_balanced.csv"
OUTPUT_PATH = "./data/tidy_confounded_balanced.csv"
META_COLS = None
MINIBATCH_SIZE = 100
CODE_SIZE = 200


if __name__ == "__main__":
    # Get sizes & meta cols
    data = pd.read_csv(INPUT_PATH)
    # TODO: when reading in the csv, squash everything into [0.0, 1.0]
    # and save all the mins & maxes so we know what range to expand them
    # back into.
    if META_COLS is None:
        META_COLS = list_categorical_columns(data)
        print "Inferred meta columns:", META_COLS
    INPUT_SIZE = len(data.columns) - len(META_COLS)
    NUM_TARGETS = len(data["Batch"].unique())

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
        # TODO: expand them back out from [0.0, 1.0] into their original ranges.
        df_adj = pd.DataFrame(adj, columns=list(range(INPUT_SIZE)))
        reformat.to_csv(
            df_adj,
            OUTPUT_PATH,
            tidy=True,
            meta_cols={
                col: data[col] for col in META_COLS
            }
        )
