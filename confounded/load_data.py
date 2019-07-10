import numpy as np
import pandas as pd

def split_features_labels(df, batch_col, meta_cols=None, sample=None):
    """Split a dataframe into features and labels numpy arrays.

    Arguments:
        df {pandas.DataFrame} -- A tidy dataframe with meta columns, a
            Batch column, and quantitative data.

    Keyword Arguments:
        meta_cols {list of strings} -- Columns that should not be
            used as features to be batch-adjusted (default:
            {["Sample", `batch_col`]})
        sample {int} -- The number of rows to sample. If None, return
            all rows. (default: {None})

    Returns:
        [(numpy.array, numpy.array)] -- Tuple of features and labels,
            where features are the quantitative data from the given
            dataframe and labels are the one-hot encoded batches for
            each instance.
    """
    if meta_cols is None:
        meta_cols = ["Sample", batch_col]
    features = np.array(df.drop(meta_cols, axis=1))
    labels = pd.get_dummies(df[batch_col])
    labels = np.array(labels, dtype=float)
    if sample is not None:
        rows = pd.DataFrame(df.index).sample(sample, replace=True)[0].values.tolist()
        features = features[rows]
        labels = labels[rows]
    return features, labels

def list_categorical_columns(df):
    """Get the names of all categorical columns in the dataframe.

    Arguments:
        df {pandas.DataFrame} -- The dataframe.

    Returns:
        list -- Names of the categorical columns in the dataframe.
    """
    return list(df.select_dtypes(exclude=['float']).columns)
