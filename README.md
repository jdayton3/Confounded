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

```bash
# TODO, see issue #19
```

## Installation

### Confounded dependencies

1. [Anaconda](https://conda.io/docs/user-guide/install/index.html) with Python 3 (*Note: the version of `h5py` that ships with Anaconda may cause some deprecation warnings.*)
2. [Tensorflow](https://www.tensorflow.org/install/)

### Benchmarking dependencies

1. [scikit-learn](http://scikit-learn.org/stable/install.html)
2. [R](https://www.r-project.org). All R scripts were run with [RStudio](https://www.rstudio.com/products/rstudio/download/), but the [knitr](https://cran.r-project.org/web/packages/knitr/index.html) package may be sufficient for running the code in the `.Rmd` files.

## Usage

To run Confounded, run the following command:

```bash
python -m src.autoencoder path/to/input_data.csv
```

To see other command line options, run:

```bash
python -m src.autoencoder -h
```

### Data preparation

Data should be a CSV in [Tidy Data](http://vita.had.co.nz/papers/tidy-data.html) format.
Additionally, the following specifications must be met:

- One column is the sample ID and is called "Sample"
- One column is the batch ID and is called "Batch"
- Any other categorical column (integer or string type) represents other "meta data"
- The rest of the columns are numeric features

For an example of properly formatted data, see `/metrics/classifiers/test_data.csv`.

### Benchmarking

Code for testing the effectiveness of Confounded is located in the `/metrics` directory.

To run the random forests classifier on the output data, edit the `INPUT_PATH` variable in `/metrics/classifiers/random_forests.py` to the path to the relevant CSV and run:

```bash
python -m metrics.classifiers.random_forests
```

To test how well the true classes are kept in the Confounded output, edit the `FILES`, `PREDICT`, and `META_COLS` values in `/metrics/classifiers/true_class.py` and run:

```bash
python -m metrics.classifiers.true_class
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
