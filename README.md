# Confounded

##### Name possibly subject to change...

## Paragraphs for proposal

When working with biological data, problems can arise from combining data from multiple sources. For example, if two researchers in separate labs quantify the same 300 RNA transcripts for lung cancer patients but use slightly different quantification methodologies, these differences can lead to bias in the data, which can confound analysis when combining data from both researchers.  This bias can be even more pronounced when using data collected from different tissue types and can skew results, such as in pan-cancer analyses.  However, recent advances in neural networks have brought to light new ways to work with multi-class data and prevent these confounding effects.

Neural networks are a tool modeled loosely after how the human brain learns; input values pass through layers of linear and nonlinear functions, the final output values are measured against objectives, and the layers of functions are adjusted to bring the outputs closer to the objectives.  This process is repeated until the outputs are sufficiently close to their targets.  Autoencoders are a type of neural network that encode and then reconstruct their input, and their traditional objective function is to construct the output as similarly to the input as possible.  Neural networks have historically had decreased effectiveness when working with data from multiple domains, in part because they may easily learn to distinguish between classes (e.g. which researcher collected the data) but then fail to discern more biologically interesting information (e.g. which gene is consistently upregulated in a disease).  Recently, researchers have experimented with discouraging the networks from distinguishing between classes by giving them [two objective functions](https://arxiv.org/pdf/1412.3474v1.pdf):  1. to learn as much as possible about the input data and 2. to forget any patterns that help distinguish between classes.  This type of dual objective function [has been used](https://arxiv.org/pdf/1511.00830.pdf) in conjunction with autoencoders; the autoencoder is "rewarded" for reconstructing the input faithfully, but "punished" for keeping class-specific patterns.  Our approach will build on this previous research by packaging this type of autoencoder to allow other researchers to remove confounding effects from multi-class data sets and to output the adjusted data for further analysis.

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