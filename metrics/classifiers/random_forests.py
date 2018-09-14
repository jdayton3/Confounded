"""Random forests classifier from scikit learn
"""

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from .csvread import CSVData

def cross_validate(path, meta_cols=None, predict="Batch", times=100, folds=5, model=RandomForestClassifier):
    classifier = model()
    print(f"Using classifier: {classifier.__class__}")
    data = CSVData(path, meta_cols, predict)
    accuracies = []
    start = time.time()
    for _ in range(times):
        print(f"{_+1}/{times}", end="\r")
        accuracies += list(cross_val_score(
            classifier, data.X, data.Y, cv=folds, scoring="accuracy"
        ))
    print()
    print(f"Time elapsed: {time.time() - start} seconds")
    return sum(accuracies) / len(accuracies)


if __name__ == "__main__":
    INPUT_PATH = "../../data/tidy_combat.csv"
    print(
        cross_validate(INPUT_PATH, times=5)
    )
