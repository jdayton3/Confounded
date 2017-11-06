# Confounded

##### Name possibly subject to change...

## Abstract

- More cool things can be done with larger data sets
- There are lot of data sets that could be combined
    - TCGA
    - LINCS
    - etc.
- Even when the exact same features are collected by multiple studies, problems come from combining data from multiple sources
    - Slightly different collection methods / instruments can lead to bias in the data
    - etc.
- We would like to apply deep learning to combine data sets and remove confounding effects
    - We will use the Python TensorFlow bindings to create an autoencoder
    - Autoencoder with dual loss function
        - Minimize the amount of change to the input data
        - Minimize the model's ability to tell between confounding classes
        - Output data the same shape & size as the input data but with confounding (is latent a synonym?) effects removed
        - Possibly output 2x data
            - The data with the confounding classes removed
            - Everything that was removed by the autoencoder
        - [Similar paper](https://arxiv.org/pdf/1705.08821.pdf) - what would we do differently?
            - It seems like they're more focused on removing hidden effects... I need to read this carefully & figure this out.
    - Create a software package (maybe using Docker) so people can remove confounding effects from their own data
    - Integrate this into [Geney](https://github.com/srp33/Geney)
- We will test our methods
    - On artificial data we construct
    - Combining MicroRNA and RNA-Seq data sets
    - Removing tissue signal from pan-cancer data


## Introduction

## Materials & Methods

## Results

## Discussion

## Conclusion