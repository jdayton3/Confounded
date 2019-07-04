import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

def split_discrete_continuous(df):
    discrete_types = ["int", "object"]
    discrete = df.select_dtypes(include=discrete_types)
    continuous = df.select_dtypes(exclude=discrete_types)
    return discrete, continuous

def mmd(sample1, sample2, kernel=rbf_kernel):
    # TODO: update the kernel to reflect the DMResNet paper:
    """ "the kernel we used is a sum of three Gaussian kernels with
    different scales...We chose the [sigma_i]s to be m/2,m,2m, where m
    is the median of the average distance between a point in the
    target sample to its nearest 25 neighbors, a popular practice in
    kernel-based methods."
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870543/
    """
    k_x_x = kernel(sample1, sample1)
    k_x_y = kernel(sample1, sample2)
    k_y_y = kernel(sample2, sample2)
    n = len(sample1)
    m = len(sample2)
    mean_xx = (1 / n**2) * k_x_x.sum()
    mean_xy = (1 / (m * n)) * k_x_y.sum()
    mean_yy = (1 / m**2) * k_y_y.sum()
    mmd_squared = mean_xx - 2 * mean_xy + mean_yy
    return mmd_squared ** 0.5

def split_into_batches(df, batch_col):
    discrete, continuous = split_discrete_continuous(df)
    batches = set(df[batch_col])
    return tuple((continuous[df[batch_col] == batch] for batch in batches))

def mmd_multi_batch(batches):
    mmds = []
    for i in range(len(batches)):
        for j in range(i+1, len(batches)):
            mmds.append(mmd(batches[i], batches[j]))
    return np.array(mmds).mean()

def calculate_mmd(df, batch_col, log_adjust=False):
    batches = split_into_batches(df, batch_col)
    if log_adjust:
        batches = tuple([log_scale(batch) for batch in batches])
    return mmd_multi_batch(batches)

def log_scale(df):
    # Get rid of negative values
    df = df.where(df.min() < 0, df - df.min())
    return np.log(df + 1.0)

class Logger(object):
    def __init__(self, metric):
        self.metric = metric
        self.values = {
            "metric": [],
            "adjuster": [],
            "dataset": [],
            "value": [],
        }

    def log(self, adjuster, dataset, value):
        self.values["metric"].append(self.metric)
        self.values["adjuster"].append(adjuster)
        self.values["dataset"].append(dataset)
        self.values["value"].append(value)

    def save(self, path):
        df = pd.DataFrame(self.values)
        df.to_csv(path, index=False)

class DataFrameCache(object):
    def __init__(self):
        self.dataframes = {} # path: dataframe

    def get_dataframe(self, path):
        try:
            return self.dataframes[path]
        except KeyError:
            self.dataframes[path] = pd.read_csv(path)
            return self.dataframes[path]

COMPARISONS_PATH = "./data/meta/comparisons.csv"

if __name__ == "__main__":
    cache = DataFrameCache()
    dataframes = cache.get_dataframe(COMPARISONS_PATH)

    logger = Logger("MMD")

    loaded_dfs = {} # path: dataframe

    for i, row in dataframes.iterrows():
        print(row.dataset)
        df = cache.get_dataframe(row["path"])
        value = calculate_mmd(df, row["batch_col"], log_adjust=("TCGA" in row.dataset))
        logger.log(row["adjuster"], row["dataset"], value)

    logger.save("./data/metrics/mmd.csv")
