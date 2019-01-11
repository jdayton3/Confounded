import numpy as np
import pandas as pd
from PIL import Image

def array2png(array, path):
    # Copied from png.py :(
    img = Image.fromarray((255 * (1.0 - array)).astype(np.uint8), 'L')
    img.save(path)

if __name__ == "__main__":
    unadjusted = pd.read_csv("./data/mnist/unadjusted.csv")
    noisy = pd.read_csv("./data/mnist/noisy.csv")
    combat = pd.read_csv("./data/mnist/noisy_combat.csv")
    confounded = pd.read_csv("./data/mnist/noisy_confounded.csv")

    dataframes = [
        unadjusted,
        noisy,
        combat,
        confounded
    ]

    names = [
        "01_unadjusted",
        "02_noisy",
        "03_combat",
        "04_confounded"
    ]

    indices = [0, 1500, 3000, 4500]

    for df, name in zip(dataframes, names):
        continuous = df.select_dtypes(exclude=["object", "int"])
        for i in indices:
            pixels = np.array(continuous.iloc[i]).reshape((28, 28))
            array2png(pixels, "./data/output/mnist/{}_{:04d}.png".format(name, i))
