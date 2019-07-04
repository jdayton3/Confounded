"""Predict the true class of MNIST digits with a SVC classifier.
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from .random_forests import cross_validate
from ..calc.mmd import COMPARISONS_PATH, DataFrameCache

class Logger(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.values = {
            "path": [],
            "model": [],
            "predict": [],
            "dataset": [],
            "col_type": [],
            "baseline": [],
            "iteration": [],
            "accuracy": [],
            "time_elapsed": [],
            "adjuster": [],
        }

    def log(self, path, model, predict_col, dataset, col_type, baseline, accuracies, times_elapsed, adjuster):
        for i, (accuracy, time) in enumerate(zip(accuracies, times_elapsed)):
            self.values["path"].append(path)
            self.values["model"].append(model.__name__)
            self.values["predict"].append(predict_col)
            self.values["dataset"].append(dataset)
            self.values["col_type"].append(col_type)
            self.values["baseline"].append(baseline)
            self.values["iteration"].append(i)
            self.values["accuracy"].append(accuracy)
            self.values["time_elapsed"].append(time)
            self.values["adjuster"].append(adjuster)

    def save(self):
        pd.DataFrame(self.values).to_csv(self.log_file, index=False)


def baseline(path, column, cache=None):
    try:
        df = cache.get_dataframe(path)
    except AttributeError:
        df = pd.read_csv(path)
    return df[column].value_counts().max() / len(df)

if __name__ == "__main__":
    cache = DataFrameCache()
    comparisons = cache.get_dataframe(COMPARISONS_PATH)
    LEARNERS = [
        (RandomForestClassifier, {"n_estimators": 10}),
        # (MLPClassifier, {"hidden_layer_sizes": tuple([5]*10), "max_iter": 1000}),
        (GaussianNB, {}),
        # (KNeighborsClassifier, {}),
        # (SVC, {"kernel": "rbf"})
    ]

    logger = Logger("./data/metrics/accuracies_sizes.csv")

    for i, row in comparisons.iterrows():
        print("Working on {}: {}".format(row["dataset"], row["adjuster"]))
        for learner in LEARNERS:
            for col_type in ["batch_col", "true_class_col"]:
                column = row[col_type]
                baseline_acc = baseline(row["path"], column, cache=cache)
                cv_scores = cross_validate(
                    row["path"],
                    predict=column,
                    model=learner[0],
                    folds=4,
                    times=3,
                    **learner[1]
                )
                logger.log(
                    row["path"],
                    learner[0],
                    column,
                    row['dataset'],
                    col_type,
                    baseline_acc,
                    cv_scores["accuracies"],
                    cv_scores["times"],
                    row["adjuster"]
                )

    logger.save()
