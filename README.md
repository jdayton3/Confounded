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

Instructions for getting started quickly with Confounded can be found on [its Docker page](https://hub.docker.com/r/jdayton3/confounded).

## Installation

The easiest way to install and run Confounded is through [its Docker image](https://hub.docker.com/r/jdayton3/confounded). If you want to install and run the source, continue reading.

TL;DR:

```bash
git clone https://github.com/jdayton3/Confounded.git
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh # Or the anaconda installer for your system.
bash Anaconda3-5.3.1-Linux-x86_64.sh # Go through the install process.
conda create -n confounded python=3.6 r-tidyverse scikit-learn
source activate confounded
pip install tensorflow # or tensorflow-gpu
```

### Confounded dependencies

1. [Anaconda](https://conda.io/docs/user-guide/install/index.html) with Python 3 (*Note: the version of `h5py` that ships with Anaconda may cause some deprecation warnings.*)
2. [Tensorflow](https://www.tensorflow.org/install/)

## Usage

To run Confounded, run the following command:

```bash
python -m confounded path/to/input_data.csv
```

To see other command line options, run:

```bash
python -m confounded -h
```

### Data preparation

Data should be a CSV in [Tidy Data](http://vita.had.co.nz/papers/tidy-data.html) format.
Additionally, the following specifications must be met:

- One column is the sample ID and is called "Sample"
- One column is the batch ID and is called "Batch"
- Any other categorical column (integer or string type) represents other "meta data"
- The rest of the columns are numeric features

For an example of properly formatted data, see `/metrics/classifiers/test_data.csv`.
