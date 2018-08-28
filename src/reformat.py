import pandas as pd

def to_csv(df, path, tidy=False, batch=None, sample=None):
    df = _add_info_cols(df, tidy, batch, sample)
    df.to_csv(path, index=False)

def _add_info_cols(df, tidy=False, batch=None, sample=None):
    if not tidy:
        df = df.T
    df['Batch'] = _check_batch(df, batch)
    df['Sample'] = _check_sample(df, sample)
    return _reorder_cols(df)

def _check_batch(df, batch):
    if not batch:
        num = int(len(df) / 2)
        return ['A'] * num + ['B'] * num
    return batch

def _check_sample(df, sample):
    if not sample:
        return df.index
    return sample

def _reorder_cols(df):
    cols = df.columns.tolist()
    meta = ["Sample", "Batch"]
    for col in meta:
        cols.remove(col)
    cols = meta + cols
    return df[cols]

if __name__ == "__main__":
    in_path = "data/mnist_matrix.csv"
    out_path = "data/tidy_mnist2.csv"
    df = pd.read_csv(in_path, header=None)
    to_csv(df, out_path)
