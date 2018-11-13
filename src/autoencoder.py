# pylint: disable=E1129,E0611,E1101
import argparse
from . import hide_warnings
import tensorflow as tf
import pandas as pd
from . import reformat
from .load_data import split_features_labels, list_categorical_columns
from .adjustments import Scaler, split_discrete_continuous
from .network import Confounded

INPUT_PATH = "./data/GSE40292_copy.csv"
OUTPUT_PATH = "./data/rna_seq_adj_test.csv"
MINIBATCH_SIZE = 5
CODE_SIZE = 20
ITERATIONS = 10000

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

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
        writer = tf.summary.FileWriter("log/small_network3", sess.graph)
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
        df_adj = pd.DataFrame(adj, columns=split_discrete_continuous(data)[-1].columns)
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
    # Setting up argparse to take in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='source-file', type=str, nargs=1,
            help='takes 1 source file for data to be passed in.')
    parser.add_argument('-o', "--output-file", type=str, nargs=1,
            help="Location for the output file.")
    parser.add_argument("-m", "--minibatch-size", type=check_positive, nargs=1,
            help="The size of the mini-batch for training. Must be positive integer.")
    parser.add_argument("-l", "--layers", type=check_positive, nargs=1,
            help="How many layers deep the autoencoder should be. Must be positive integer.")

    args = parser.parse_args()

    # Adding user options to code
    INPUT_PATH = args.file[0]
    if args.output_file:
        OUTPUT_PATH = args.output_file[0]
    else:
        OUTPUT_PATH = INPUT_PATH.rstrip(".csv") + "_confounded.csv"
    if args.minibatch_size:
        MINIBATCH_SIZE = args.minibatch_size[0]
    if args.layers:
        CODE_SIZE = args.layers[0]
    autoencoder(INPUT_PATH, OUTPUT_PATH, MINIBATCH_SIZE, CODE_SIZE, ITERATIONS)
