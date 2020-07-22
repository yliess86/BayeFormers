<p align="center">
    <img style="max-width: 648px;" alt="Logo" src=BayeFormers.png />
</p>

<p align="center">
    <a href="https://github.com/yliess86/BayeFormers/blob/master/LICENSE">
        <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
    </a>
    <a href="https://www.python.org/downloads/release/python-360/">
        <img alt="Python 3.6+" src="https://img.shields.io/badge/python-3.6+-green.svg" />
    </a>
    <a href="https://pytorch.org/">
        <img alt="Pytorch 1.4+" src="https://img.shields.io/badge/pytorch-1.4.0+-blue.svg" />
    </a>
</p>

<p align="center">
General API for <b>Deep Bayesian Variational Inference by Backpropagation</b>.<br />The repository has been designed to work with <b>Transformers</b> like architectures.<br />Compatible with the <a href="https://huggingface.co/">HuggingFace Transformers</a> models.
</p>



## Setup

Installation of the required python libraries is done through pip.

```bash
$ cd BayeFormers
$ (sudo) pip3 install -r requirements.txt
```

## Usage

```python
from bayeformers import to_bayesian

import bayeformers.nn as bnn
import torch
import torch.nn as nn
import torch.nn.functional as F


# Frequentist Model Definition
class Model(nn.Module):
    pass


# Train Frequentist Model
model = Model()

predictions = model(inputs)
loss = F.nll(inputs, labels, reduction="sum")

# Turn Frequentist Model to Bayesian Model (MOPED Initializatipn)
bayesian_model = to_bayesian(model, delta=0.05, freeze=True)

# Train Bayesian Model
predictions = torch.zeros(samples, batch_size, *output_dim)
log_prior = torch.zeros(samples, batch_size)
log_variational_posterior = torch.zeros(samples, batch_size)

for s in samples:
    predictions[s] = bayesian_model(inputs)
    log_prior[s] = bayesian_model.log_prior()
    log_variational_posterior[s] = bayesian_model.log_variational_posterior()

predictions = predictions.mean(0)
log_prior = log_prior.mean(0)
log_variational_posterior = log_variational_posterior.mean(0)

nll = F.nll(predictions, labels, reduction="sum")
loss = (log_variational_posterior - log_prior) / n_batches + nll
```

## Examples

```bash
$ python3 -m examples.mlp_mnist
$ python3 -m examples.bert_glue --help
$ python3 -m examples.bert_squad --help
```

## References

**Libraries**
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/)

**Papers**
- "Weight Uncertainty in Neural Networks", Blundell et al., ICML 2015, [Arxiv](https://arxiv.org/abs/1505.05424)
- "Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes", Krishnan et al., AAAI 2020, [Arxiv](https://arxiv.org/abs/1906.05323v3)

**Articles**
- "Bayesian inference: How we are able to chase the Posterior", Ritchie Vink, [Blog](https://www.ritchievink.com/blog/2019/06/10/bayesian-inference-how-we-are-able-to-chase-the-posterior/)
- "Weight Uncertainty in Neural Networks", Nitarshan Rajkumar, [Blog](https://www.nitarshan.com/bayes-by-backprop/)