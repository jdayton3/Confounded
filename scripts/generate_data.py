import numpy as np
import pandas as pd
import argparse

def generate_data(rows_per_batch=10, batches=2, columns=3, centers=None, variances=None):
    return pd.concat(
        (
            pd.DataFrame(
                ["A"] * rows_per_batch + ["B"] * rows_per_batch,
                columns=["Batch"]
            ),
            pd.DataFrame(
                np.concatenate(
                    (
                        np.random.normal(0.0, 1.0, (rows_per_batch, 3)),
                        np.random.normal(100.0, 2.0, (rows_per_batch, 3))
                    )
                )
            )
        ),
        axis=1
    ).reset_index().rename(columns={"index": "Sample"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate some batched data.")
    parser.add_argument("--n_batches", "-b", type=int, nargs=1, help="The number of batches to generate", default=2)
    parser.add_argument("--n_rows", "-r", type=int, nargs=1, help="The number of rows per batch", default=100)
    parser.add_argument("--n_cols", "-c", type=int, nargs=1, help="The number of quantitative columns", default=3)
    args = parser.parse_args()
    print(generate_data(args.n_rows, args.n_batches, args.n_cols))
