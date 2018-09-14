import numpy as np
import pandas as pd
import gzip

class RNASeq(object):
    def __init__(self):
        self.data = self.load_data()
        self.cur_ix = 0

    def load_data(self):
        data = []
        with gzip.open("./data/RNASeq.txt.gz", "r") as infile:
            header = infile.readline()
            for line in infile:
                line = line.strip("\n").split("\t")[1:]
                line = [float(x) for x in line]
                data.append(line)
        return np.array(data)

    def next_batch(self, batch_size):
        if batch_size > len(self.data):
            raise ValueError("Argument `batch_size` greater than length of dataset.")
        if self.cur_ix + batch_size > len(self.data):
            np.random.shuffle(self.data)
            self.cur_ix = 0
        batch = self.data[self.cur_ix:self.cur_ix + batch_size]
        self.cur_ix += batch_size
        return batch

def split_features_labels(df, meta_cols=None):
    """Split a dataframe into features and labels numpy arrays.

    Arguments:
        df {pandas.DataFrame} -- A tidy dataframe with meta columns, a
            Batch column, and quantitative data.

    Keyword Arguments:
        meta_cols {list of strings} -- Columns that should not be
            used as features to be batch-adjusted (default:
            {["Sample", "Batch"]})

    Returns:
        [(numpy.array, numpy.array)] -- Tuple of features and labels,
            where features are the quantitative data from the given
            dataframe and labels are the one-hot encoded batches for
            each instance.
    """
    if meta_cols is None:
        meta_cols = ["Sample", "Batch"]
    features = np.array(df.drop(meta_cols, axis=1))
    labels = np.array(pd.get_dummies(df["Batch"]), dtype=float)
    return features, labels

if __name__ == "__main__":
    data = RNASeq()
    print(data.data.shape)
