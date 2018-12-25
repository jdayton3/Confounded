"""Random forests classifier from scikit learn
"""

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from .csvread import CSVData

def cross_validate(path, meta_cols=None, predict="Batch", times=100, folds=5, model=RandomForestClassifier, **kwargs):
    classifier = model(**kwargs)
    data = CSVData(path, meta_cols, predict)
    accuracies = []
    elapsed_times = []
    for _ in range(times):
        start = time.time()
        try:
            accuracies += list(cross_val_score(
                classifier, data.X, data.Y, cv=folds, scoring="accuracy"
            ))
            elapsed_times.append(time.time() - start)
        except ValueError as e:
            print("Something didn't work with file {}, column {}, and classifier {}:".format(
                path, predict, model.__name__)
            )
            print(str(e))
    return {
        "accuracies": accuracies,
        "times": elapsed_times
    }

if __name__ == "__main__":
    INPUT_PATH = "../../data/tidy_combat.csv"
    print(
        cross_validate(INPUT_PATH, times=5)
    )
