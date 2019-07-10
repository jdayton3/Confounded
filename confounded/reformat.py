import pandas as pd

def to_csv(df, path, tidy=False, meta_cols=None):
    """Save a dataframe as a CSV, and optionally add meta columns and
        transpose the dataframe.

    Arguments:
        df {pandas.DataFrame} -- The data to be saved.
        path {str} -- The path where the csv should be saved.

    Keyword Arguments:
        tidy {bool} -- True if the rows are instances and the columns
            are features; False otherwise. (default: {False})
        meta_cols {dict} -- Dictionary of {column_name: [values, ...]}
            (default: {None})
    """
    if not tidy:
        df = df.T
    df = _add_meta_cols(df, meta_cols)
    df.to_csv(path, index=False)

def _add_meta_cols(df, meta_cols):
    if meta_cols is None:
        meta_cols = {}
    sample = meta_cols.get("Sample")
    df['Sample'] = _check_sample(df, sample)
    for col_name, col in meta_cols.items():
        if col_name in ["Sample"]:
            continue
        df[col_name] = col
    return _reorder_cols(df, list(meta_cols.keys()))

def _check_sample(df, sample):
    if sample is None:
        return df.index
    return sample

def _reorder_cols(df, meta_cols):
    cols = df.columns.tolist()
    for col in meta_cols:
        cols.remove(col)
    cols = meta_cols + cols
    return df[cols]

if __name__ == "__main__":
    import functools
    import operator
    in_path = "data/mnist_matrix.csv"
    out_path = "data/tidy_mnist2.csv"
    df = pd.read_csv(in_path, header=None)
    to_csv(df, out_path, tidy=False, meta_cols={
        "Batch": None,
        "Sample": None,
        "Digit": functools.reduce(
            operator.add,
            [[x] * 1000 for x in range(10)]
        )
    })
