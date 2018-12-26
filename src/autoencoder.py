# pylint: disable=E1129,E0611,E1101
import argparse
from time import time
from . import hide_warnings
import tensorflow as tf
import pandas as pd
from . import reformat
from .load_data import split_features_labels, list_categorical_columns
from .adjustments import Scaler, split_discrete_continuous
from .network import Confounded

MINIBATCH_SIZE = 100
CODE_SIZE = 2000
ITERATIONS = 5000
DISCRIMINATOR_LAYERS = 2
LOG_FILE = "./data/metrics/training.csv"
ACTIVATION = tf.nn.relu
BATCH_COL = "Batch"

def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

class SummaryLogger(object):
    def __init__(self, log_file, code_size, d_layers, minibatch_size, activation, batch_col):
        self.start_time = time()
        self.log_file = log_file
        self.code_size = code_size
        self.d_layers = d_layers
        self.minibatch_size = minibatch_size
        self.activation = activation
        self.batch_col = batch_col
        self.values = {
            "start_time": [],
            "batch_column": [],
            "minibatch_size": [],
            "code_size": [],
            "discriminator_layers": [],
            "activation": [],
            "time": [],
            "iteration": [],
            "ae_loss": [],
            "disc_loss": [],
            "dual_loss": [],
        }

    def log(self, iteration, ae_loss, disc_loss, dual_loss):
        self.values["start_time"].append(self.start_time)
        self.values["batch_column"].append(self.batch_col)
        self.values["minibatch_size"].append(self.minibatch_size)
        self.values["code_size"].append(self.code_size)
        self.values["discriminator_layers"].append(self.d_layers)
        self.values["activation"].append(".".join([
            self.activation.__module__,
            self.activation.__name__
        ]))
        self.values["time"].append(time())
        self.values["iteration"].append(iteration)
        self.values["ae_loss"].append(ae_loss)
        self.values["disc_loss"].append(disc_loss)
        self.values["dual_loss"].append(dual_loss)

    def save(self):
        new_df = pd.DataFrame(self.values)
        old_df = None
        try:
            old_df = pd.read_csv(self.log_file)
        except:
            pass
        if old_df is not None:
            new_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
        new_df.to_csv(self.log_file, index=False)

def autoencoder(input_path,
                output_path,
                minibatch_size=100,
                code_size=200,
                iterations=10000,
                d_layers=2,
                activation=tf.nn.relu,
                batch_col="Batch"):
    # Get sizes & meta cols
    data = pd.read_csv(input_path)
    scaler = Scaler()
    data = scaler.squash(data)
    meta_cols = list_categorical_columns(data)
    print("Inferred meta columns:", meta_cols)
    input_size = len(data.columns) - len(meta_cols)
    num_targets = len(data[batch_col].unique())

    c = Confounded(input_size, code_size, num_targets, discriminator_layers=d_layers, activation=activation)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/arbitrary_batch", sess.graph)
        tf.global_variables_initializer().run()

        logger = SummaryLogger(
            LOG_FILE,
            CODE_SIZE,
            DISCRIMINATOR_LAYERS,
            MINIBATCH_SIZE,
            ACTIVATION,
            BATCH_COL
        )
        # Train
        for i in range(iterations):
            features, labels = split_features_labels(
                data,
                batch_col,
                meta_cols=meta_cols,
                sample=minibatch_size
            )
            summary, disc_loss, ae_loss, dual_loss, _, _, _ = sess.run([
                merged,
                c.d_loss,
                c.mse,
                c.loss,
                c.outputs,
                c.optimizer,
                c.d_optimizer
            ], feed_dict={
                c.inputs: features,
                c.targets: labels,
            })
            writer.add_summary(summary, i)
            logger.log(i, ae_loss, disc_loss, dual_loss)

        logger.save()

        # Run the csv through confounded
        features, labels = split_features_labels(data, batch_col, meta_cols=meta_cols)
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
            help="How many layers deep the discriminator should be. Must be positive integer.")
    parser.add_argument("-b", "--batch-col", type=str, nargs=1,
            help="Which column contains the batch to adjust for.")
    parser.add_argument("-c", "--code-size", type=check_positive, nargs=1,
            help="How many nodes in the code layer of the autoencoder.")

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
        DISCRIMINATOR_LAYERS = args.layers[0]
    if args.batch_col:
        BATCH_COL = args.batch_col[0]
    if args.code_size:
        CODE_SIZE = args.code_size[0]

    autoencoder(
        INPUT_PATH,
        OUTPUT_PATH,
        MINIBATCH_SIZE,
        CODE_SIZE,
        ITERATIONS,
        d_layers=DISCRIMINATOR_LAYERS,
        activation=ACTIVATION,
        batch_col=BATCH_COL
    )
