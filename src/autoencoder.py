# pylint: disable=E1129,E0611,E1101

from . import hide_warnings
import tensorflow as tf
import pandas as pd
from . import reformat
from .load_data import split_features_labels, list_categorical_columns
from .network import Confounded

INPUT_PATH = "./data/GSE40292_copy.csv"
OUTPUT_PATH = "./data/rna_seq_adj.csv"
META_COLS = None
MINIBATCH_SIZE = 100
CODE_SIZE = 200
ITERATIONS = 10000

def _split_discrete_continuous(df):
    discrete_types = ['object', 'int']
    discrete = df.select_dtypes(include=discrete_types)
    continuous = df.select_dtypes(exclude=discrete_types)
    return discrete, continuous

class Scaler(object):
    """Scale or unscale a dataframe from [min, max] <-> [0, 1]
    """
    def __init__(self):
        self.col_min = None
        self.col_max = None

    def squash(self, df):
        """Adjust the dataframe to the [0, 1] range.

        Arguments:
            df {pandas.DataFrame} -- The quantitative dataframe to be
                squashed.

        Returns:
            pandas.DataFrame -- The squashed dataframe.
        """
        discrete, continuous = _split_discrete_continuous(df)

        self.col_min = continuous.min()
        self.col_max = continuous.max()
        scaled = (continuous - self.col_min) / (self.col_max - self.col_min)

        return pd.concat([discrete, scaled], axis="columns")

    def unsquash(self, df):
        """Adjust the dataframe back to the original range.

        Arguments:
            df {pandas.DataFrame} -- The quantitative dataframe to be
                expanded.

        Returns:
            pandas.DataFrame -- The dataframe with each column expanded
                to its original range.
        """
        if self.col_min is None or self.col_max is None:
            raise Exception(
                "Error: Scaler.squash() must be run "
                "before Scaler.unsquash() can be used."
            )
        discrete, continuous = _split_discrete_continuous(df)
        scaled = continuous * (self.col_max - self.col_min) + self.col_min
        if discrete.size == 0:
            return scaled
        return pd.concat([discrete, scaled], axis="columns")


def autoencoder(input_path, output_path, minibatch_size=100, code_size=200, iterations=10000):
    # Get sizes & meta cols
    data = pd.read_csv(input_path)
    scaler = Scaler()
    data = scaler.squash(data)
    meta_cols = list_categorical_columns(data)
    print("Inferred meta columns:", meta_cols)
    input_size = len(data.columns) - len(meta_cols)
    num_targets = len(data["Batch"].unique())

    c = Confounded(input_size, code_size, num_targets)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/rna_seq_scaled", sess.graph)
        tf.global_variables_initializer().run()

        # Train
        for i in range(iterations):
            features, labels = split_features_labels(
                data,
                meta_cols=meta_cols,
                sample=minibatch_size
            )
            summary, disc, out, _ = sess.run([merged, c.outputs, c.optimizer, c.d_optimizer], feed_dict={
                c.inputs: features,
                c.targets: labels,
            })
            writer.add_summary(summary, i)

        # Run the csv through confounded
        features, labels = split_features_labels(data, meta_cols=meta_cols)
        adj, = sess.run([c.outputs], feed_dict={
            c.inputs: features,
            c.targets: labels,
        })
        # Save adjusted & non-adjusted numbers
        df_adj = pd.DataFrame(adj, columns=_split_discrete_continuous(data)[-1].columns)
        df_adj = scaler.unsquash(df_adj)
        reformat.to_csv(
            df_adj,
            output_path,
            tidy=True,
            meta_cols={
                col: data[col] for col in meta_cols
            }
        )

if __name__ == "__main__":
    autoencoder(INPUT_PATH, OUTPUT_PATH, MINIBATCH_SIZE, CODE_SIZE, ITERATIONS)
