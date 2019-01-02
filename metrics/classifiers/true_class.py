"""Predict the true class of MNIST digits with a SVC classifier.
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from .random_forests import cross_validate

class Logger(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.values = {
            "path": [],
            "model": [],
            "predict": [],
            "baseline": [],
            "iteration": [],
            "accuracy": [],
            "time_elapsed": [],
        }

    def log(self, path, model, predict_col, baseline, accuracies, times_elapsed):
        for i, (accuracy, time) in enumerate(zip(accuracies, times_elapsed)):
            self.values["path"].append(path)
            self.values["model"].append(model.__name__)
            self.values["predict"].append(predict_col)
            self.values["baseline"].append(baseline)
            self.values["iteration"].append(i)
            self.values["accuracy"].append(accuracy)
            self.values["time_elapsed"].append(time)

    def save(self):
        pd.DataFrame(self.values).to_csv(self.log_file, index=False)


def baseline(path, column):
    df = pd.read_csv(path)
    return df[column].value_counts().max() / len(df)

if __name__ == "__main__":
    PATHS = [
        # "data/avery/GSE37199/clinical.csv",
        # "data/avery/GSE37199/clinical_confounded.csv",
        # "data/avery/GSE37199/tidy_combat.csv",
        "data/mnist/noisy.csv",
        "data/mnist/noisy_combat.csv",
        "data/output/mnist/mnist.csv",
        "../Modified-BatchEffectRemoval/output.csv"
    ]
    LEARNERS = [
        (RandomForestClassifier, {}),
        # (MLPClassifier, {"hidden_layer_sizes": tuple([5]*10), "max_iter": 1000}),
        # (GaussianNB, {}),
        # (KNeighborsClassifier, {}),
        # (SVC, {"kernel": "rbf"})
    ]
    PREDICT = [
        'Batch',
        'Digit'
        # 'centre',
        # 'plate',
        # 'replicate',
        # 'Stage',
    ]

    logger = Logger("./data/metrics/accuracies.csv")
    for path in PATHS:
        for column in PREDICT:
            try:
                baseline_acc = baseline(path, column)
            except KeyError:
                print("Column {} not in file {}. Skipping.".format(column, path))
                continue
            for learner in LEARNERS:
                cv_scores = cross_validate(
                    path,
                    predict=column,
                    model=learner[0],
                    times=5,
                    **learner[1]
                )
                logger.log(
                    path,
                    learner[0],
                    column,
                    baseline_acc,
                    cv_scores["accuracies"],
                    cv_scores["times"]
                )
    logger.save()
