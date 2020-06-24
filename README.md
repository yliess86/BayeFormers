# Bayeformers

General API for deep bayesian variationnal inference.
Idealy designed to work with transformers like architectures.

## API

```python
from bayeformers import bayesian
from bayeformers import to_bayesian
from transformers import BERT

import torch.nn as nn
import torch.nn.functional as F
import bayeformers.nn as bnn

@bayesian
class BERT_bayesian(nn.Module):
    pass

BERT_bayesian = to_bayesian(BERT)

bnn.GaussianParameter
bnn.GaussianMixtureParameter

bnn.Sequential
bnn.ModuleList

bnn.Linear
bnn.Conv1d
bnn.Conv2d
bnn.Conv3d
bnn.DeConv1d
bnn.DeConv2d
bnn.DeConv3d

bnn.KLD

bnn.Entropy
bnn.AleatoricUncertainty
bnn.EpistemicUncertainty
```

## Details

```python
from torch.distributions.normal import Normal


def reparametrization_trick(eps: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return mu + eps * std


class GaussianParameter(nn.Module):
    def __init__(mu: torch.Tensor, rho: torch.Tensor, size: torch.Size) -> None:
        super(GaussianParameter, self).__init__()
        self.mu, self.rho = nn.Parameter(mu), nn.Parameter(rho)
        self.size = size
        self.distrib = Normal(0, 1)

    def sample() -> torch.Tensor:
        return reparametrization_trick(self.distrib.sample(self.size), self.mu, F.softplus(self.rho))

    def log_prob() -> torch.Tensor:
        return self.distrib.log_prob()


class Linear(bnn.BayesianModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.W = GaussianParameter(mu, rho, torch.Size((out_features, in_features)))
        if bias:
            self.b = GaussianParameter(mu, rho, torch.Size((out_features, )))
        else:
            self.b = self.register_parameter("b", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        W, b = self.W.sample(), None if self.b is None else self.Wb.sample()
        return inputs @ W + b


class BayesianModule:
    @property
    def kld(self) -> torch.Tensor:
        pass

    @property
    def log_prob(self) -> torch.Tensor:
        pass
```