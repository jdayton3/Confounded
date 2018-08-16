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


if __name__ == "__main__":
    from . import png
    arr = png.png2array("~/Downloads/mnist_png/training/5/0.png")
    noiser1 = Noise(arr.shape, order=2)
    noiser2 = Noise(arr.shape, order=2)
    png.array2png(noiser1.adjust(arr), "new1.png")
    png.array2png(noiser2.adjust(arr), "new2.png")
