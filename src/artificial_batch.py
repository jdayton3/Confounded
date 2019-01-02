import pandas as pd
import numpy as np
from random import seed
from .adjustments import Noise, split_discrete_continuous
from .png import array2png

if __name__ == "__main__":
    seed(0)
    df = pd.read_csv("./data/tidy_nonadjusted.csv")
    discrete, continuous = split_discrete_continuous(df)
    n_pixels = len(continuous.columns)
    n1 = Noise((n_pixels,), order=10, discount_factor=0.02)
    n2 = Noise((n_pixels,), order=10, discount_factor=0.02)
    batch = []
    images = []
    for i, row in continuous.iterrows():
        add_noise = np.random.uniform(-0.1, 0.0, size=(n_pixels,))
        mult_noise = np.random.uniform(0.85, 0.95, size=(n_pixels,))
        image = row.tolist()
        image = (image + add_noise) * mult_noise
        if i % 2 == 0:
            image = n1.adjust(image)
            batch.append("A")
        else:
            image = n2.adjust(image)
            batch.append("B")
        images.append(image)
    noisy = pd.DataFrame(images)
    stuff = pd.concat([discrete, noisy], axis="columns")
    stuff["Batch"] = batch
    stuff.to_csv("./data/mnist/noisy.csv", index=False)
    # x1 = images[0]
    # array2png(np.array(x1).reshape((28, 28)), "thing.png")
