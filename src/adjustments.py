"""Synthetic batch effects"""
import numpy as np

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
        def __init__(self, shape, discount_factor, activation=np.tanh):
            self.weights = np.random.normal(size=shape)
            self.bias = np.random.normal()
            self.discount_factor = discount_factor
            self.activation = activation

        def adjust(self, array):
            return array + (
                self.activation(
                    array * self.weights + self.bias
                ) * self.discount_factor
            )
