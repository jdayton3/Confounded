# Confounded

##### Name possibly subject to change...

## Paragraphs for proposal

When working with biological data, problems can arise from combining data from multiple sources. If two separate researchers generate the same type of data with all the same variables for a particular observation, subtle differences in collection methods can lead to bias in the data, which can confound attempts at accurate analysis when combining the data from both researchers.  This bias can be even more pronounced when using data collected with from different tissue types and can skew results, such as in pan-cancer analyses.  However, recent advances in neural networks have made it possible to eliminate these class-specific confounding effects.

Neural networks are a tool for machine learning in which input values pass through layers of linear and nonlinear functions. The values that are output by neural networks are measured with objective functions, and the layers of functions are adjusted to bring the outputs closer to the objectives.  This process is repeated until the outputs are sufficiently close to their targets.  Autoencoders are a type of neural network that encode and then reconstruct their input, and their traditional objective function is to construct the output as close to the input as possible.  

Neural networks have historically had decreased effectiveness when working with data from multiple domains, in part because they may easily learn to distinguish between classes (e.g. which researcher collected the data) but then fail to discern more biologically interesting information (e.g. which gene is consistently upregulated in a disease).  Our proposed project is to use neural networks to remove confounding effects from multi-class data sets by creating an autoencoder that takes input with multiple known classes and adjusts the data.  This autoencoder will have two objective functions:  1. to reconstruct input as faithfully as possible and 2. to remove any class-specific patterns from the data.  This type of dual objective function [has been used](https://arxiv.org/pdf/1412.3474v1.pdf) to allow a neural network to learn meaningful representations in data while ignoring differences caused by class groupings and [has also been used](https://arxiv.org/pdf/1511.00830.pdf) in conjunction with autoencoders.  Our approach would build on previous research by using an autoencoder with this dual objective function to remove confounding effects from multi-class data sets and to output the adjusted data for further analysis.  We will package our software to allow other researchers to remove confounding effects from their own data.

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
    - Combining microarray and RNA-Seq data sets
    - Removing tissue signal from pan-cancer data


### Previous research:

- Citation
    - What they did
    - How their research supports our concept (deep learning is the right approach because other people have used deep learning)
    - How their research fell short 
    - How our research will overcome their shortcomings
- [Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/abs/1412.3474v1)
    - ...
- [Causal Effect Inference with Deep Latent-Variable Models](https://arxiv.org/pdf/1705.08821.pdf)
    - ...
- [The Variational Fair Autoencoder](https://arxiv.org/pdf/1511.00830.pdf)
    - ...


## Introduction

## Materials & Methods

## Results

## Discussion

## Conclusion