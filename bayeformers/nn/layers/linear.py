from bayeformers.nn.parameters.base import NoneParameter
from bayeformers.nn.parameters.base import Parameter
from bayeformers.nn.parameters.gaussian import Gaussian
from bayeformers.nn.parameters.gaussian import DEFAULT_SCALED_GAUSSIAN_MIXTURE
from bayeformers.nn.parameters.initializations import DEFAULT_UNIFORM
from bayeformers.nn.parameters.initializations import Initialization
from torch import Size
from torch import Tensor
from torch.nn import Module
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, bias: Optional[bool] = True,
        initialization: Optional[Initialization] = DEFAULT_UNIFORM,
        prior: Optional[Parameter] = DEFAULT_SCALED_GAUSSIAN_MIXTURE
    ) -> None:
        super(Linear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.initialization = initialization

        size = Size((self.out_features, self.in_features))
        self.weight = Gaussian(size, self.initialization)
        self.weight_prior = prior
        
        if bias:
            size = Size((self.out_features, ))
            self.bias = Gaussian(size, self.initialization)
            self.bias_prior = prior
        else:
            self.bias = NoneParameter()
            self.bias_prior = NoneParameter()

        self.log_prior = 0.0
        self.log_variational_posterior = 0.0

    def forward(self, input: Tensor) -> Tensor:
        weight, bias = self.weight.sample(), self.bias.sample() 
        
        self.log_prior = self.weight_prior.log_prob(weight)
        self.log_prior += self.bias_prior.log_prob(bias)
        self.log_variational_posterior = self.weight.log_prob(weight)
        self.log_variational_posterior += self.bias.log_prob(bias)
        
        return F.linear(input, weight, bias=bias)