"""Random forests classifier from scikit learn
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from csvread import CSVData

def cross_validate(path, times=100, folds=5):
    classifier = RandomForestClassifier()
    data = CSVData(path)
    accuracies = []
    for _ in range(times):
        accuracies += list(cross_val_score(
            classifier, data.X, data.Y, cv=folds, scoring="accuracy"
        ))
    return sum(accuracies) / len(accuracies)


if __name__ == "__main__":
    print(
        cross_validate("test_data.csv")
    )
