# Confounded

**Name possibly subject to change...**

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
  - Note: Really cool before/after PCAish figures at the end of the article
- [Cool Reddit summary of VAEs vs GANs](https://www.reddit.com/r/MachineLearning/comments/4r3pjy/variational_autoencoders_vae_vs_generative/)
  - ...
- [Source Separation](https://en.wikipedia.org/wiki/Blind_signal_separation) - maybe not super relevant
  - ...
- [Extracting a Biologically Relevant Latent Space from Cancer Transcriptomes with Variational Autoencoders](https://www.biorxiv.org/content/early/2017/10/02/174474)
  - Interesting... Seems like this may support the idea that autoencoders CAN generate biologically realistic transcriptomes.
- [Privacy-preserving generative deep neural networks support clinical data sharing](https://www.biorxiv.org/content/early/2017/07/05/159756.1)
  - Similar to what we're trying to do, but with an emphasis on preserving privacy and using GANs instead of autoencoders.  In other words, I think they are generating new fake data from the clinical latent space & then using the fake data in analyses.

## Introduction

## Materials & Methods

### Materials

- Tensorflow
  - Wide use in Deep Learning community--76,000 stars and 37,000 watchers on [GitHub](https://github.com/tensorflow/tensorflow), which is much more than [Keras](https://github.com/fchollet/keras), [Theano](https://github.com/Theano/Theano), [Lasagne](https://github.com/Lasagne/Lasagne), [Caffe](https://github.com/BVLC/caffe), and [Torch](https://github.com/torch/torch7).
  - [Developed by Google](https://www.tensorflow.org/) - associated with a big, stable company.  Probably will be around for a long time.
  - Abstracts away the really low-level stuff like backpropagation, but allows access to important low-level stuff that needs tweaked
    - Graph structures
    - Loss functions
  - Fast.
    - For development
      - Write in Python (high-level programming language)
      - Great built-in visualization library, which makes for easier debugging
    - For running
      - Once the graph is created, it runs in C++ / CUDA, which is important because Deep Learning algorithms are very computationally intensive
- Docker
  - Easier for other scientists to download and run because the container will have dependencies built in
    - Working with GPUs is possible but takes more steps to [get it set up](https://medium.com/@gooshan/for-those-who-had-trouble-in-past-months-of-getting-google-s-tensorflow-to-work-inside-a-docker-9ec7a4df945b)
  - Multi-platform
    - Easier development, can focus on features instead of on developing multiple code bases for multiple operating systems
  - Containerization means it won't clash with other programs 
  - The rest of Geney's services also run in Docker (?) so we can keep it running with a smaller team (I think this is a good reason to use Docker, but grant people may not think so)

## Results

## Discussion

## Conclusion