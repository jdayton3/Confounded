# Confounded

Batch adjustment with adversarial autoencoders.

## Quick Start

## Installation

### Confounded dependencies

1. [Anaconda](https://conda.io/docs/user-guide/install/index.html)
2. [Tensorflow](https://www.tensorflow.org/install/)

### Benchmarking dependencies

1. [scikit-learn](http://scikit-learn.org/stable/install.html)
2. [R](https://www.r-project.org). All R scripts were run with [RStudio](https://www.rstudio.com/products/rstudio/download/), but the [knitr](https://cran.r-project.org/web/packages/knitr/index.html) package may be sufficient for running the code in the `.Rmd` files.

## Usage

```bash
python2 -m src.autoencoder
```

### Data preparation

### Interpreting results

### Benchmarking

## More information

### Comparison to other methods

Results were obtained by creating a balanced CSV of artificially batched MNIST images, using various methods to correct for the artificial batch effects, and classifying based on batch using the scikit-learn RandomForestClassifier with default parameters.

Higher accuracies represent that the classifier was still able to detect batch effects after adjustment. Since the two batches were balanced, a perfect batch adjustment should yield a classification accuracy of 0.5.

| Type of adjustment | Accuracy |   Time  |
|:-------------------|---------:|--------:|
| None               |      1.0 |  3.615s |
| ComBat             |      1.0 |  3.825s |
| Confounded         |  0.50564 | 46.538s |

### Goals and Timeline

[The project prospectus](Prospectus.pdf) has information about previous research and goals for the project.
