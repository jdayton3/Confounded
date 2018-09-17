# Confounded

## Batch adjustment with adversarial autoencoders

### Why use batch adjustment?

Data are biased based on the way they are collected.
When analyzing data from multiple sources, that bias can mess up the results, so it is often useful to remove source-based bias, or "batch effects."

### Why use Confounded?

Most batch adjusters assume that batch effects are linear and that source bias in one variable doesn't affect bias in other variables.
However, modern analysis tools like machine learning are really good at learning nonlinear relationships, so if even very small nonlinear effects still exist after correction, modern analysis can still be biased by those effects.
(See the [comparison to other methods](comparison-to-other-methods) for more info.)

Confounded uses deep neural networks to identify and remove both linear *and* nonlinear batch effects.

### How does it work?

Confounded uses two neural networks to adjust data for batch effects.
One network (the discriminator) looks at the data and learns to tell between batches, and the other network (the autoencoder) makes small tweaks to the data in order to "fool" the discriminator.
The autoencoder also tries to keep the adjusted data as similar as possible to the original data.
This process continues until the discriminator can't distinguish the batches and the autoencoder is faithfully reproducing the data without batch effects.

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

To alter Confounded's behavior, adjust the following variables in `/src/autoencoder.py`:

| Variable         | Description                                                                                                                                                              |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `INPUT_PATH`     | The path to the input data.                                                                                                                                              |
| `OUTPUT_PATH`    | The path where the adjusted data should be stored.                                                                                                                       |
| `META_COLS`      | A list of columns that should be treated as meta  data, not as features to be adjusted                                                                                   |
| `INPUT_SIZE`     | The number of features to be adjusted. (See  [#16](https://github.com/jdayton3/Confounded/issues/16))                                                                    |
| `NUM_TARGETS`    | The number of targets for the discriminator to  distinguish between (i.e. the number of distinct  batches, see  [#16](https://github.com/jdayton3/Confounded/issues/16)) |
| `MINIBATCH_SIZE` | The size of the  [mini-batch](https://datascience.stackexchange.com/q/16807)  for training.                                                                              |
| `CODE_SIZE`      | The size of the encoding layer of the autoencoder.                                                                                                                       |

*Note: issues [#12](https://github.com/jdayton3/Confounded/issues/12) and [#16](https://github.com/jdayton3/Confounded/issues/16) are open and will make setting these variables easier.*

### Data preparation

Data should be a CSV in [Tidy Data](http://vita.had.co.nz/papers/tidy-data.html) format.
Additionally, the following specifications must be met:

- One column is the sample ID and is called "Sample"
- One column is the batch ID and is called "Batch"
- The rest of the columns are numeric features

For an example of properly formatted data, see `/metrics/classifiers/test_data.csv`.

*Note: issue [#18](https://github.com/jdayton3/Confounded/issues/18) will allow for more "meta columns" to be included in the input data.*

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

Higher accuracies represent that the classifier was still able to detect batch effects after adjustment.
Since the two batches were balanced, a perfect batch adjustment should yield a classification accuracy of 0.5.
We also interpret a longer training time to mean that the classifier had a more difficult time detecting differences between batches.

| Type of adjustment | Accuracy |   Time  |
|:-------------------|---------:|--------:|
| None               |      1.0 |  3.615s |
| ComBat             |      1.0 |  3.825s |
| Confounded         |  0.50564 | 46.538s |

#### ComBat

We used the [ComBat](https://doi.org/10.1093/biostatistics/kxj037) implementation from the [R sva package](https://www.bioconductor.org/packages/release/bioc/html/sva.html).
Our code to do this is in `src/r/combat.Rmd` and can be run using [RStudio](https://www.rstudio.com/).

### Goals and Timeline

[The project prospectus](Prospectus.pdf) has information about previous research and goals for the project.
