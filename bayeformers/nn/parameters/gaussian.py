from bayeformers.nn.functionnal import gaussian_log_prob
from bayeformers.nn.functionnal import scaled_gaussian_mixture_log_prob
from bayeformers.nn.functionnal import reparametrization_trick
from bayeformers.nn.parameters.base import parameter
from bayeformers.nn.parameters.base import Parameter
from bayeformers.nn.parameters.initializations import DEFAULT_UNIFORM
from bayeformers.nn.parameters.initializations import Initialization
from torch import Size
from torch import Tensor
from torch.distributions.normal import Normal
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class Gaussian(Parameter):
    def __init__(
        self, size: Size,
        initialization: Optional[Initialization] = DEFAULT_UNIFORM,
        dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
        super(Gaussian, self).__init__()
        self.size, self.dtype = size, dtype
        self.initialization = initialization
        self.mu = parameter(self.size, dtype=self.dtype)
        self.rho = parameter(self.size, dtype=self.dtype)
        self.normal = Normal(0, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mu, self.rho = self.initialization(self.mu, self.rho)

    @property
    def sigma(self) -> Tensor:
        return F.softplus(self.rho)

    def sample(self) -> Tensor:
        eps = self.normal.sample(self.size).to(self.mu.device)
        return reparametrization_trick(eps, self.mu, self.sigma)

    def log_prob(self, input: Tensor) -> Tensor:
        return gaussian_log_prob(input, self.mu, self.sigma)


class ScaledGaussianMixture(Parameter):
    def __init__(self, pi: float, sigma1: float, sigma2: float) -> None:
        super(ScaledGaussianMixture, self).__init__()
        self.pi = pi
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.gaussian1 = Normal(0, self.sigma1)
        self.gaussian2 = Normal(0, self.sigma2)

    def sample(self) -> Tensor:
        return 0.0

    def log_prob(self, input: Tensor) -> Tensor:
        return scaled_gaussian_mixture_log_prob(
            input,
            self.gaussian1.log_prob(input),
            self.gaussian2.log_prob(input),
            self.pi
        )


DEFAULT_SCALED_GAUSSIAN_MIXTURE = ScaledGaussianMixture(
    0.5, np.exp(-0), np.exp(-6)
)