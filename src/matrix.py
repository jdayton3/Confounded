#!/usr/bin/env python3

from . import png
from .adjustments import Noise
import numpy as np
import pandas as pd
import glob

def pngs2matrix(directory):
    paths = glob.glob(directory + "/**/*.png", recursive=True)
    n_pixels = len(png.png2array(paths[0]).flatten())
    n_pngs = len(paths)
    noiser1 = Noise((n_pixels,), order=2)
    noiser2 = Noise((n_pixels,), order=2)
    # ComBat needs n_probes x n_samples
    matrix = np.ndarray((n_pixels, n_pngs))
    for i, path in enumerate(paths):
        pixels = png.png2array(path).flatten()
        if i < n_pngs / 2:
            pixels = noiser1.adjust(pixels)
        else:
            pixels = noiser2.adjust(pixels)
        matrix[:, i] = pixels
    return matrix

def unflatten(array):
    return np.reshape(array, (28, 28))

if __name__ == "__main__":
    directory = '/home/jdayton3/Downloads/mnist_png/testing'
    matrix = pngs2matrix(directory)
    pd.DataFrame(matrix).to_csv("./data/mnist_matrix.csv", index=False, header=False)
