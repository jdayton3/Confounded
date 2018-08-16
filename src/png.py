# coding: utf-8
from PIL import Image
import numpy as np
from imageio import imread
import glob

def array2png(array, path):
    img = Image.fromarray((255 * (1.0 - array)).astype(np.uint8), 'L')
    img.save(path)

def png2array(path):
    return 1.0 - imread(path) / 255

if __name__ == "__main__":
    directory = '/home/jdayton3/Downloads/mnist_png/testing/**/*.png'
    w, h = 28, 28
    data = np.zeros((h, w)) + 0.5
    data[3, 3] = 0.0
    data[4, 4] = 1.0
    array2png(data, "my.png")
    arr = png2array("my.png")
