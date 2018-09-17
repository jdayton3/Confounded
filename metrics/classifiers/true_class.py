"""Predict the true class of MNIST digits with a SVC classifier.
"""

from sklearn.svm import SVC
from .random_forests import cross_validate

if __name__ == "__main__":
    FILES = ["nonadjusted", "batches2", "confounded_digit", "batches_balanced", "confounded_balanced"]
    PREDICT = "Digit"
    META_COLS = ["Sample", "Batch", "Digit"]
    for name in FILES:
        print(name)
        path = f"./data/tidy_{name}.csv"
        print(
            cross_validate(
                path,
                meta_cols=META_COLS,
                predict=PREDICT
                model=SVC, # SVC takes about 40 minutes to run.
                times=5
            )
        )
