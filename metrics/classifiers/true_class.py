"""Predict the true class of MNIST digits with a SVC classifier.
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .random_forests import cross_validate

def baseline(path, column):
    df = pd.read_csv(path)
    return df[column].value_counts().max() / len(df)

if __name__ == "__main__":
    # FILES = ["nonadjusted", "batches2", "confounded_digit", "batches_balanced", "confounded_balanced"]
    # PATHS = [f'./data/tidy_{name}.csv' for name in FILES]
    PATHS = [
        # "data/avery/GSE25507/tidy_confounded.csv",
        "data/avery/GSE37199/tidy.csv",
        "data/avery/GSE37199/tidy_confounded.csv",
        "data/avery/GSE37199/tidy_combat.csv",
        # "data/tidy_batches_balanced_combat.csv",
        # "data/avery/GSE39582/tidy.csv",
        # "data/avery/GSE40292/tidy.csv",
    ]
    PREDICT = "Batch"
    for path in PATHS:
        print(path)
        print("Baseline: {}".format(baseline(path, PREDICT)))
        print(
            cross_validate(
                path,
                predict=PREDICT,
                # model=SVC, # SVC takes about 40 minutes to run on MNIST.
                model=RandomForestClassifier,
                times=5
            )
        )
