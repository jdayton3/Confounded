# Confounded

Batch adjustment with adversarial autoencoders.

## Quick Start

## Installation

## Usage

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
