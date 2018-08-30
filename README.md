# Confounded

Batch adjustment with adversarial autoencoders.

## Quick Start

## Installation

### Confounded dependencies

1. [Anaconda](https://conda.io/docs/user-guide/install/index.html). *Note: Python 2 is currently required for Confounded, but Python 3 is used for benchmarking scripts. [An issue](https://github.com/jdayton3/Confounded/issues/1) is open to upgrade all code to the latest version of Python 3.*
2. [Tensorflow](https://www.tensorflow.org/install/)

### Benchmarking dependencies

1. [scikit-learn](http://scikit-learn.org/stable/install.html)
2. [R](https://www.r-project.org). All R scripts were run with [RStudio](https://www.rstudio.com/products/rstudio/download/), but the [knitr](https://cran.r-project.org/web/packages/knitr/index.html) package may be sufficient for running the code in the `.Rmd` files.

## Usage

To run Confounded, run the following command:

```bash
python2 -m src.autoencoder
```

It is currently only set up to run on artificially batched MNIST digits, but [an issue](https://github.com/jdayton3/Confounded/issues/13) is open to allow it to run on any properly formatted CSV.

### Data preparation

Confounded isn't currently set up to take data from any source besides MNIST.

In future iterations, data will need to follow the specifications listed in [this README](metrics/README.md).

### Benchmarking

Code for testing the effectiveness of Confounded is located in the `/metrics` directory.

To run the random forests classifier on the output data, edit the `INPUT_PATH` variable in `/metrics/classifiers/random_forests.py` to the path to the relevant CSV and run:

```bash
$ python3 random_forests.py
5/5  # all 5 iterations of cross-validation have completed.
0.75 # the classifier had 75% accuracy.
```

## More information

### Comparison to other methods

Results were obtained by creating a balanced CSV of artificially batched MNIST images, using various methods to correct for the artificial batch effects, and classifying based on batch using the scikit-learn RandomForestClassifier with default parameters.

Higher accuracies represent that the classifier was still able to detect batch effects after adjustment. Since the two batches were balanced, a perfect batch adjustment should yield a classification accuracy of 0.5. We also interpret a longer training time to mean that the classifier had a more difficult time detecting differences between batches.

| Type of adjustment | Accuracy |   Time  |
|:-------------------|---------:|--------:|
| None               |      1.0 |  3.615s |
| ComBat             |      1.0 |  3.825s |
| Confounded         |  0.50564 | 46.538s |

#### ComBat

We used the [ComBat](https://doi.org/10.1093/biostatistics/kxj037) implementation from the [R sva package](https://www.bioconductor.org/packages/release/bioc/html/sva.html). Our code to do this is in `src/r/combat.Rmd` and can be run using [RStudio](https://www.rstudio.com/).

### Goals and Timeline

[The project prospectus](Prospectus.pdf) has information about previous research and goals for the project.
