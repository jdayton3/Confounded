# pylint: disable=E1129,E0611,E1101
import argparse
import datetime
import random
from tqdm import tqdm
from . import hide_warnings
import tensorflow as tf
import pandas as pd
from . import reformat
from .load_data import split_features_labels, list_categorical_columns
from .adjustments import Scaler, SigmoidScaler, split_discrete_continuous
from .network import Confounded


def parse_arguments():
    activation = tf.nn.relu

    def positive_int(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
        return ivalue

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file', metavar='source-file', type=str,
        help='Path to input file.')
    parser.add_argument(
        '-o', "--output-file", type=str,
        help="Path to output file.")
    parser.add_argument(
        "-m", "--minibatch-size", type=positive_int, default=100,
        help="The size of the mini-batch for training. Must be positive integer.")
    parser.add_argument(
        "-l", "--layers", type=positive_int, default=10,
        help="How many layers deep the discriminator should be. Must be positive integer.")
    parser.add_argument(
        "-a", "--ae-layers", type=positive_int, default=2,
        help="How many layers in each of the encoding and decoding portions of the autoencoder.")
    parser.add_argument(
        "-b", "--batch-col", type=str, default="Batch",
        help="Which column contains the batch to adjust for.")
    parser.add_argument(
        "-c", "--code-size", type=positive_int, default=20,
        help="How many nodes in the code layer of the autoencoder.")
    parser.add_argument(
        "-e", "--early-stopping", type=positive_int, default=None,
        help="How many iterations without improvement before stopping early. Default: None")
    parser.add_argument(
        "-s", "--scaling", choices=["linear", "sigmoid"], default="linear",
        help="Type of scaling to perform on the input data.")
    parser.add_argument(
        "-w", "--loss-weight", type=float, default=1.0,
        help="Weight applied to the discriminator loss when training the autoencoder.")
    parser.add_argument(
        "-f", "--log-file", type=str, default="./data/metrics/log.csv",
        help="Path to file to log results.")
    parser.add_argument(
        "-i", "--iterations", type=positive_int, default=10000,
        help="Number of iterations of minibatches to run.")
    parser.add_argument(
        "-d", "--save-model", type=str, default="",
        help="Path to save the model weights. Weights are not saved if path is not specified.")
    parser.add_argument(
        "-r", "--load-model", type=str, default="",
        help="Path to model weights checkpoint to load."
            " Weights are initialized randomly if path is not specified.")

    args = parser.parse_args()

    if not args.output_file:
        args.output_file = args.file.rstrip(".csv") + "_confounded.csv"
    args.activation = activation

    return args

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
                save_weights_path,
                load_weights_path,
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

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars_to_replace = [var for var in all_vars if "do_not_save" in var.name]
    vars_to_keep = [var for var in all_vars if var not in vars_to_replace]
    saver = tf.train.Saver(vars_to_keep)

    with tf.Session() as sess:
        if load_weights_path:
            saver.restore(sess, load_weights_path)
            print("Model loaded from path: {}".format(load_weights_path))
            tf.variables_initializer(vars_to_replace).run()
        else:
            tf.global_variables_initializer().run()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/{}".format(now()), sess.graph)

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

        print("Training Confounded")
        for i in tqdm(range(iterations)):
            features, labels = split_features_labels(
                data,
                batch_col,
                meta_cols=meta_cols,
                sample=minibatch_size
            )
            summary, disc_loss, ae_loss, dual_loss, _, _, _ = sess.run([
                merged,
                c.d_loss,
                c.ae_loss,
                c.loss,
                c.outputs,
                c.optimizer,
                c.d_optimizer,
            ], feed_dict={
                c.inputs: features,
                c.targets: labels,
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
        if save_weights_path:
            saver.save(sess, save_weights_path)
            print("Model saved in path: {}".format(save_weights_path))

        print("Adjusting the input data")
        features, labels = split_features_labels(data, batch_col, meta_cols=meta_cols)
        adj, = sess.run([c.outputs], feed_dict={
            c.inputs: features,
            c.targets: labels,
        })
        print("Saving data to {}".format(output_path))
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

    args = parse_arguments()

    autoencoder(
        args.file,
        args.output_file,
        args.save_model,
        args.load_model,
        args.minibatch_size,
        args.code_size,
        args.iterations,
        d_layers=args.layers,
        ae_layers=args.ae_layers,
        activation=args.activation,
        batch_col=args.batch_col,
        early_stopping=args.early_stopping,
        scaling=args.scaling,
        disc_weighting=args.loss_weight,
        log_file=args.log_file
    )
