"""Adjustments to dataframes"""
import numpy as np
import pandas as pd

def split_discrete_continuous(df):
    discrete_types = ['object', 'int']
    discrete = df.select_dtypes(include=discrete_types)
    continuous = df.select_dtypes(exclude=discrete_types)
    return discrete, continuous

class Scaler(object):
    """Scale or unscale a dataframe from [min, max] <-> [0, 1]
    """
    def __init__(self):
        self.col_min = None
        self.col_max = None
        self.squashed = False

    def squash(self, df):
        """Adjust the dataframe to the [0, 1] range.

        Arguments:
            df {pandas.DataFrame} -- The quantitative dataframe to be
                squashed.

        Returns:
            pandas.DataFrame -- The squashed dataframe.
        """
        discrete, continuous = split_discrete_continuous(df)
        scaled = self._squash_continuous(continuous)
        self.squashed = True
        return pd.concat([discrete, scaled], axis="columns")

    def _squash_continuous(self, continuous):
        self.col_min = continuous.min()
        self.col_max = continuous.max()
        already_in_range = (
            (self.col_min >= 0.0) &
            (self.col_min <= 1.0) &
            (self.col_max >= 0.0) &
            (self.col_max <= 1.0)
        )
        self.col_min = np.where(already_in_range, 0.0, self.col_min)
        self.col_max = np.where(already_in_range, 1.0, self.col_max)

        scaled = (continuous - self.col_min) / (self.col_max - self.col_min)
        return scaled

    def unsquash(self, df):
        """Adjust the dataframe back to the original range.

        Arguments:
            df {pandas.DataFrame} -- The quantitative dataframe to be
                expanded.

        Returns:
            pandas.DataFrame -- The dataframe with each column expanded
                to its original range.
        """
        if not self.squashed:
            raise Exception(
                "Error: Scaler.squash() must be run "
                "before Scaler.unsquash() can be used."
            )
        discrete, continuous = split_discrete_continuous(df)
        scaled = self._unsquash_continuous(continuous)
        if discrete.size == 0:
            return scaled
        return pd.concat([discrete, scaled], axis="columns")

    def _unsquash_continuous(self, continuous):
        scaled = continuous * (self.col_max - self.col_min) + self.col_min
        return scaled

class SigmoidScaler(Scaler):
    """Scaler class that scales continuous values into (0.0, 1.0) using
    the sigmoid function.
    """
    def _squash_continuous(self, continuous):
        return self.__sigmoid(continuous)

    def _unsquash_continuous(self, continuous):
        return self.__logit(continuous)

    def __sigmoid(self, continuous):
        return 1.0 / (1.0 + np.exp(-continuous))

    def __logit(self, continuous):
        """inverse of sigmoid"""
        return np.log(continuous) - np.log(1 - continuous)
