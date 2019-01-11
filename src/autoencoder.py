# pylint: disable=E1129,E0611,E1101
import argparse
import datetime
import random
from . import hide_warnings
import tensorflow as tf
import pandas as pd
from . import reformat
from .load_data import split_features_labels, list_categorical_columns
from .adjustments import Scaler, SigmoidScaler, split_discrete_continuous
from .network import Confounded

MINIBATCH_SIZE = 100
CODE_SIZE = 2000
ITERATIONS = 1000
DISCRIMINATOR_LAYERS = 10
AUTOENCODER_LAYERS = 2
LOG_FILE = "./data/metrics/log.csv"
ACTIVATION = tf.nn.relu
BATCH_COL = "plate"
EARLY_STOPPING = None
SCALING = "linear" # or "sigmoid"
LOSS_WEIGHTING = 1.0

def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def now():
    """Return the current time in iso format"""
    return datetime.datetime.now().isoformat()

def should_train_dual(i, tot_iterations):
    """For a given iteration, should we train ae & disc?

    Cools down over time.
    """
    probability = 1.0 - i / tot_iterations
    probability = 0.1
    return random.uniform(0.0, 1.0) < probability

class SummaryLogger(object):
    def __init__(self,
                 log_file,
                 input_path,
                 output_path,
                 code_size,
                 d_layers,
                 ae_layers,
                 minibatch_size,
                 activation,
                 batch_col,
                 scaling,
                 loss_weight):
        self.start_time = now()
        self.log_file = log_file
        self.input_path = input_path
        self.output_path = output_path
        self.code_size = code_size
        self.d_layers = d_layers
        self.ae_layers = ae_layers
        self.minibatch_size = minibatch_size
        self.activation = activation
        self.batch_col = batch_col
        self.scaling = scaling
        self.loss_weight = loss_weight
        self.values = {
            "start_time": [],
            "input_path": [],
            "output_path": [],
            "batch_column": [],
            "minibatch_size": [],
            "code_size": [],
            "discriminator_layers": [],
            "autoencoder_layers": [],
            "activation": [],
            "scaling_method": [],
            "loss_weight": [],
            "time": [],
            "iteration": [],
            "ae_loss": [],
            "disc_loss": [],
            "dual_loss": [],
        }

    def log(self, iteration, ae_loss, disc_loss, dual_loss):
        self.values["start_time"].append(self.start_time)
        self.values["input_path"].append(self.input_path)
        self.values["output_path"].append(self.output_path)
        self.values["batch_column"].append(self.batch_col)
        self.values["minibatch_size"].append(self.minibatch_size)
        self.values["code_size"].append(self.code_size)
        self.values["discriminator_layers"].append(self.d_layers)
        self.values["autoencoder_layers"].append(self.ae_layers)
        self.values["activation"].append(".".join([
            self.activation.__module__,
            self.activation.__name__
        ]))
        self.values["scaling_method"].append(self.scaling)
        self.values["loss_weight"].append(self.loss_weight)
        self.values["time"].append(now())
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
                ae_layers=2,
                activation=tf.nn.relu,
                batch_col="Batch",
                early_stopping=None,
                scaling="linear",
                disc_weighting=1.0,
                log_file="log.csv"):
    # Get sizes & meta cols
    data = pd.read_csv(input_path)
    scaling_options = {
        "linear": Scaler,
        "sigmoid": SigmoidScaler
    }
    scaler = scaling_options[scaling]()
    data = scaler.squash(data)
    meta_cols = list_categorical_columns(data)
    print("Inferred meta columns:", meta_cols)
    input_size = len(data.columns) - len(meta_cols)
    num_targets = len(data[batch_col].unique())

    c = Confounded(
        input_size,
        code_size,
        num_targets,
        discriminator_layers=d_layers,
        autoencoder_layers=ae_layers,
        activation=activation,
        disc_weghting=disc_weighting
    )

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/bladder-figs", sess.graph)
        tf.global_variables_initializer().run()

        logger = SummaryLogger(
            log_file,
            input_path,
            output_path,
            code_size,
            d_layers,
            ae_layers,
            minibatch_size,
            activation,
            batch_col,
            scaling,
            disc_weighting
        )
        # Train
        n_since_improvement = 0
        best_loss = float("inf")

        sequential_iterations = iterations * 2
        for i in range(sequential_iterations):
            features, labels = split_features_labels(
                data,
                batch_col,
                meta_cols=meta_cols,
                sample=minibatch_size
            )
            if i < iterations / 2:
                optimizer = c.ae_optimizer
            elif i < iterations:
                optimizer = c.d_optimizer
            else:
                optimizer = c.optimizer
            summary, disc_loss, ae_loss, dual_loss, _, _ = sess.run([
                merged,
                c.d_loss,
                c.mse,
                c.loss,
                c.outputs,
                optimizer,
            ], feed_dict={
                c.inputs: features,
                c.targets: labels,
            })

            if i > iterations or should_train_dual(i, sequential_iterations):
                sess.run([c.optimizer, c.d_optimizer], feed_dict={
                    c.inputs: features, c.targets: labels
                })

            writer.add_summary(summary, i)
            logger.log(i, ae_loss, disc_loss, dual_loss)
            if dual_loss < best_loss:
                n_since_improvement = 0
                best_loss = dual_loss
            else:
                n_since_improvement += 1
            if early_stopping is not None and n_since_improvement >= early_stopping:
                print("No loss improvement for {} iterations. Stopping at iteration {}.".format(
                    n_since_improvement,
                    i
                ))
                break

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
    parser.add_argument("-a", "--ae-layers", type=check_positive, nargs=1,
            help="How many layers in each of the encoding and decoding portions of the autoencoder.")
    parser.add_argument("-b", "--batch-col", type=str, nargs=1,
            help="Which column contains the batch to adjust for.")
    parser.add_argument("-c", "--code-size", type=check_positive, nargs=1,
            help="How many nodes in the code layer of the autoencoder.")
    parser.add_argument("-e", "--early-stopping", type=check_positive, nargs=1,
            help="How many iterations without improvement before stopping early. Default: None")
    parser.add_argument("-s", "--scaling", choices=["linear", "sigmoid"],
            help="Type of scaling to perform on the input data.")
    parser.add_argument("-w", "--loss-weight", type=float, nargs=1,
            help="Weight applied to the discriminator loss when training the autoencoder.")
    parser.add_argument("-f", "--log-file", type=str, nargs=1,
            help="Path to file to log results.")

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
    if args.ae_layers:
        AUTOENCODER_LAYERS = args.ae_layers[0]
    if args.batch_col:
        BATCH_COL = args.batch_col[0]
    if args.code_size:
        CODE_SIZE = args.code_size[0]
    if args.early_stopping:
        EARLY_STOPPING = args.early_stopping[0]
    if args.scaling:
        SCALING = args.scaling
    if args.loss_weight:
        LOSS_WEIGHTING = args.loss_weight[0]
    if args.log_file:
        LOG_FILE = args.log_file[0]

    autoencoder(
        INPUT_PATH,
        OUTPUT_PATH,
        MINIBATCH_SIZE,
        CODE_SIZE,
        ITERATIONS,
        d_layers=DISCRIMINATOR_LAYERS,
        ae_layers=AUTOENCODER_LAYERS,
        activation=ACTIVATION,
        batch_col=BATCH_COL,
        early_stopping=EARLY_STOPPING,
        scaling=SCALING,
        disc_weighting=LOSS_WEIGHTING,
        log_file=LOG_FILE
    )
