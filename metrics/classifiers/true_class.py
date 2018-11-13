"""Predict the true class of MNIST digits with a SVC classifier.
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .random_forests import cross_validate

if __name__ == "__main__":
    # FILES = ["nonadjusted", "batches2", "confounded_digit", "batches_balanced", "confounded_balanced"]
    # PATHS = [f'./data/tidy_{name}.csv' for name in FILES]
    PATHS = [
        "./data/dist_matching/tidy.csv",
        "./data/dist_matching/tidy_confounded.csv",
        "./data/dist_matching/tidy_combat.csv",
    ]
    PREDICT = "Batch"
    for path in PATHS:
        print(path)
        print(
            cross_validate(
                path,
                predict=PREDICT,
                # model=SVC, # SVC takes about 40 minutes to run on MNIST.
                model=RandomForestClassifier,
                times=5
            )
        )
