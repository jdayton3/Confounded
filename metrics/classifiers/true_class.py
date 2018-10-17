"""Predict the true class of MNIST digits with a SVC classifier.
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .random_forests import cross_validate

if __name__ == "__main__":
    # FILES = ["nonadjusted", "batches2", "confounded_digit", "batches_balanced", "confounded_balanced"]
    # PATHS = [f'./data/tidy_{name}.csv' for name in FILES]
    PATHS = ["./data/GSE40292_copy.csv", "./data/rna_seq_adj.csv"]
    PREDICT = "Clinical_diagnosis"
    for path in PATHS:
        print(path)
        print(
            cross_validate(
                path,
                predict=PREDICT,
                # model=SVC, # SVC takes about 40 minutes to run.
                model=RandomForestClassifier,
                times=5
            )
        )
