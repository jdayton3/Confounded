# pylint: disable=E1129,E0611,E1101
import csv
import sys
import argparse
from . import hide_warnings
import tensorflow as tf
import pandas as pd
from . import reformat
from .load_data import split_features_labels, list_categorical_columns
from .network import Confounded

INPUT_PATH = "./src/data/"
OUTPUT_PATH = "./src/data/tidy_confounded_balanced.csv"
META_COLS = None
MINIBATCH_SIZE = 100
CODE_SIZE = 200

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

if __name__ == "__main__":

    # Setting up argparse to take in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='Source_File', type=str, nargs=1,
        help='takes 1 source file for data to be passed in') 
    parser.add_argument("-c", "--Meta_Cols", type=str, nargs='*', 
            help="A list of columns to be treated as meta data. Defaults to all columns w/out floating point data.")
    parser.add_argument("-m", "--Minibatch_Size", type=check_positive, nargs=1, 
            help="The size of the mini-batch for training. Must be positive integer.")
    parser.add_argument("-l", "--Layers", type=check_positive, nargs=1, 
        help="How many layers deep the autoencoder should be. Must be positive integer.")

    args = parser.parse_args()

    # Adding user options to code
    INPUT_PATH += args.file[0]
    if args.Meta_Cols:
        META_COLS = args.Meta_Cols
        # Checking that user input matches columns in the CSV if they don't program terminates
        with open(INPUT_PATH, "r") as f:
            reader = csv.reader(f)
            columnNames = next(reader)
        for title in META_COLS:
            if title not in columnNames:
                print("COLUMNS GIVEN DO NOT MATCH")
                sys.exit()
    if args.Minibatch_Size:
        MINIBATCH_SIZE = args.Minibatch_Size[0]
    if args.Layers:
        CODE_SIZE = args.Layers[0]

    # Get sizes & meta cols
    data = pd.read_csv(INPUT_PATH)
    if META_COLS is None:
        META_COLS = list_categorical_columns(data)
        print(("Inferred meta columns:", META_COLS))
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
