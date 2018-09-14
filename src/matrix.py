#!/usr/bin/env python3

from . import png
from .adjustments import Noise
from . import reformat
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
    nonadjusted = np.ndarray((n_pixels, n_pngs))
    adjusted = np.ndarray((n_pixels, n_pngs))
    for i, path in enumerate(paths):
        pixels = png.png2array(path).flatten()
        nonadjusted[:, i] = pixels
        if i % 2 == 0:
            pixels = noiser1.adjust(pixels)
        else:
            pixels = noiser2.adjust(pixels)
        adjusted[:, i] = pixels
    return nonadjusted, adjusted, paths

def unflatten(array):
    return np.reshape(array, (28, 28))

if __name__ == "__main__":
    directory = '/home/jdayton3/Downloads/mnist_png/testing'
    nonadjusted, adjusted, paths = pngs2matrix(directory)
    df = pd.DataFrame(adjusted)
    df.to_csv("./data/mnist_matrix.csv", index=False, header=False)

    df2 = pd.DataFrame(nonadjusted)
    reformat.to_csv(df2, "./data/tidy_nonadjusted.csv", meta_cols={
        "Batch": ['A'] * 10000,
        "Sample": df.T.index,
        "Digit": [path.split('/')[-2] for path in paths]
    })

    reformat.to_csv(df, "./data/tidy_batches_balanced.csv", meta_cols={
        "Batch": ['A', 'B'] * 5000,
        "Sample": df.T.index,
        "Digit": [path.split('/')[-2] for path in paths]
    })
