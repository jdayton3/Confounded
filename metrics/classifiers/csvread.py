import pandas as pd
from sklearn.model_selection import train_test_split

class CSVData(object):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.labels = df["Batch"]
        self.features = df.drop(["Batch", "Sample"], axis="columns")
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.features, self.labels, test_size=0.2
        )

    @property
    def X(self):
        return self.features

    @property
    def Y(self):
        return self.labels

if __name__ == "__main__":
    data = CSVData("test_data.csv")
    print(data.X)
    print()
    print(data.Y)
