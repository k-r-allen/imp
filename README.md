# Infinite Mixture Prototypes
## Kelsey Allen, Evan Shelhamer, Hanul Shin, Josh Tenenbaum

## Abstract
We propose infinite mixture prototypes to adaptively represent both simple and complex data distributions for few-shot learning. Infinite mixture prototypes combine deep representation learning with Bayesian nonparametrics, representing each class by a set of clusters, unlike existing prototypical methods that represent each class by a single cluster. By inferring the number of clusters, infinite mixture prototypes interpolate between nearest neighbor and prototypical representations in a learned feature space, which improves accuracy and robustness in the few-shot regime. We show the importance of adaptive capacity for capturing complex data distributions such as super-classes (like alphabets in character recognition), with 10-25% absolute accuracy improvements over prototypical networks, while still maintaining or improving accuracy on standard few-shot learning benchmarks. By clustering labeled and unlabeled data with the same rule, infinite mixture prototypes achieve state-of-the-art semi-supervised accuracy, and can perform purely unsupervised clustering, unlike existing fully- and semi-supervised prototypical methods.

## Link to paper
http://proceedings.mlr.press/v97/allen19b.html

## Code
This repository is adapted from https://github.com/renmengye/few-shot-ssl-public for PyTorch 0.3.1

### Installation

We use Python 2.7.13. Other versions may work with some modifications.
To install requirements:
```
pip install -r requirements.txt
```

### Usage Examples
`submit_omniglot.sh` provides example usage of the main file.

We also have submission scripts for running code on a slurm cluster. 
Please refer to `submit_all_models.sh` and `submit_super.sh` for examples.
