"""Synthetic batch effects"""
import numpy as np
import pandas as pd

class Noise(object):
    def __init__(
            self, shape, order=2, discount_factor=0.01, activation=np.tanh
        ):
        self._layers = [
            self._Layer(shape, discount_factor, activation)
            for _ in range(order)
        ]

    def adjust(self, x):
        y = x
        for layer in self._layers:
            y = layer.adjust(y)
        return y

    class _Layer(object):
        def __init__(self, shape, discount_factor, activation=np.tanh, min_=0.0, max_=1.0):
            self.weights = np.random.normal(size=shape)
            self.bias = np.random.normal()
            self.discount_factor = discount_factor
            self.activation = activation
            self.max = max_
            self.min = min_

        def adjust(self, array):
            adjusted = array + (
                self.activation(
                    array * self.weights + self.bias
                ) * self.discount_factor
            )
            return self.threshold(adjusted)

        def threshold(self, array):
            too_low = array < self.min
            too_high = array > self.max
            array[too_low] = self.min
            array[too_high] = self.max
            return array

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

    def squash(self, df):
        """Adjust the dataframe to the [0, 1] range.

        Arguments:
            df {pandas.DataFrame} -- The quantitative dataframe to be
                squashed.

        Returns:
            pandas.DataFrame -- The squashed dataframe.
        """
        discrete, continuous = split_discrete_continuous(df)

        self.col_min = continuous.min()
        self.col_max = continuous.max()
        scaled = (continuous - self.col_min) / (self.col_max - self.col_min)

        return pd.concat([discrete, scaled], axis="columns")

    def unsquash(self, df):
        """Adjust the dataframe back to the original range.

        Arguments:
            df {pandas.DataFrame} -- The quantitative dataframe to be
                expanded.

        Returns:
            pandas.DataFrame -- The dataframe with each column expanded
                to its original range.
        """
        if self.col_min is None or self.col_max is None:
            raise Exception(
                "Error: Scaler.squash() must be run "
                "before Scaler.unsquash() can be used."
            )
        discrete, continuous = split_discrete_continuous(df)
        scaled = continuous * (self.col_max - self.col_min) + self.col_min
        if discrete.size == 0:
            return scaled
        return pd.concat([discrete, scaled], axis="columns")



if __name__ == "__main__":
    from . import png
    arr = png.png2array("~/Downloads/mnist_png/training/5/0.png")
    noiser1 = Noise(arr.shape, order=2)
    noiser2 = Noise(arr.shape, order=2)
    png.array2png(noiser1.adjust(arr), "new1.png")
    png.array2png(noiser2.adjust(arr), "new2.png")
