import pandas as pd
import numpy as np
from .mmd import split_discrete_continuous, Logger, DataFrameCache, COMPARISONS_PATH

def calculate_mse(df1, df2):
    _, genes1 = split_discrete_continuous(df1)
    _, genes2 = split_discrete_continuous(df2)
    squared_error = (np.array(genes1) - np.array(genes2))**2
    return squared_error.mean()

if __name__ == "__main__":
    cache = DataFrameCache()
    comparisons = cache.get_dataframe(COMPARISONS_PATH)
    logger = Logger("MSE")

    for i, row in comparisons.iterrows():
        df = cache.get_dataframe(row["path"])
        original = cache.get_dataframe(row["unadjusted"])
        value = calculate_mse(df, original)
        logger.log(row["adjuster"], row["dataset"], value)

    logger.save("./data/metrics/mse.csv")
