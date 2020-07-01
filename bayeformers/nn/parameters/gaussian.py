# -*- coding: utf-8 -*-
"""bayeformers.nn.parameters.gaussian

The files provides classes and methods for gaussian related bayesian
parameters. Guassian and ScaledGaussianMixture are available. 
"""
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
    """Gaussian

    Gaussian Parameter parametrized by mu and rho (std is softplus of rho,
    this enables stability and forces the value to be positive). Actual value
    of the parameter is computed using the reparametrization trick. Weight
    is sampled by sampling a normal distribution centered on 0 with std of 1.
    The sammple value epsilon is then scaled using the mu and rho values.

    eps ~ N(0, 1)
    std = log(1 + exp(rho))
    W = mu + eps * std

    Attributes:
        size (Size): size of the parameter, weight
        initialization (Optional[Initialization]): initilization callback,
            responsible for the weight initialization. Initialises mu and rho.
            {default: DEFAULT_UNIFORM}
        mu (nn.Parameter): mu parameter of the gaussian
        rho (nn.Parameter): rho parameter of the gaussian
        normal (Normal): normal distribution to sample epsilon
    """

    def __init__(
        self, size: Size,
        initialization: Optional[Initialization] = DEFAULT_UNIFORM,
        dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
        """Initilization

        Arguments:
            size (Size): size of the parameter, weight
        
        Keyword Arguments:
            initialization (Optional[Initialization]): initilization callback,
                responsible for the weight initialization.
                Initialises mu and rho. {default: DEFAULT_UNIFORM}
            dtype (Optional[torch.dtype]): data type of the parameter
                {default: torch.float32}
        """
        super(Gaussian, self).__init__()
        self.size, self.dtype = size, dtype
        self.initialization = initialization
        self.mu = parameter(self.size, dtype=self.dtype)
        self.rho = parameter(self.size, dtype=self.dtype)
        self.normal = Normal(0, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset Parameter

        Reset mu and rho values using the initialization callback.
        """
        self.mu, self.rho = self.initialization(self.mu, self.rho)

    @property
    def sigma(self) -> Tensor:
        """Sigma

        Returns:
            Tensor: sigma return as the sofplus of rho
        """
        return F.softplus(self.rho)

    def sample(self) -> Tensor:
        """Sample

        Reparamtrization trick allowing for differentiable sampling.
        eps ~ N(0, 1)
        W = mu + eps * std

        Returns:
            Tensor: sampled gaussian weight using the reparametrization trick.
        """
        eps = self.normal.sample(self.size).to(self.mu.device)
        return reparametrization_trick(eps, self.mu, self.sigma)

    def log_prob(self, input: Tensor) -> Tensor:
        """Gaussian Log Probability

        Arguments:
            input (Tensor): sampled value of the gaussian weight

        Returns:
            Tensor: log probability
        """
        return gaussian_log_prob(input, self.mu, self.sigma)


class ScaledGaussianMixture(Parameter):
    """Scaled Gaussian Mixture

    Scaled Mixture of Gaussians.
    Do not compute samples as this distribution is only used for the weight
    priors.

    Attributes:
        pi (float): interpolation factor between the two gaussians basis
        sigma1 (float): sigma for the first gaussian
        sigma2 (float): sigma for the second gaussian
        gaussian1 (Normal): normal distribution for the first gaussian
        gaussian2 (Normal): normal distribution for the second gaussian
    """

    def __init__(self, pi: float, sigma1: float, sigma2: float) -> None:
        """Initialize

        Arguments:
            pi (float): interpolation factor between the two gaussians basis
            sigma1 (float): sigma for the first gaussian
            sigma2 (float): sigma for the second gaussian
        """
        super(ScaledGaussianMixture, self).__init__()
        self.pi = pi
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.gaussian1 = Normal(0, self.sigma1)
        self.gaussian2 = Normal(0, self.sigma2)

    def sample(self) -> Tensor:
        """Sample

        Is not implemented for now for reasons stated above. Thus returns 0.0
        for the moment.
        """
        return 0.0

    def log_prob(self, input: Tensor) -> Tensor:
        """Scale Gaussian Mixture Log Probability

        Arguments:
            input (Tensor): sampled value of the gaussian weight

        Returns:
            Tensor: log probability
        """
        return scaled_gaussian_mixture_log_prob(
            input,
            self.gaussian1.log_prob(input),
            self.gaussian2.log_prob(input),
            self.pi
        )


"""Default initialization for the Scale Gaussian Mixture"""
DEFAULT_SCALED_GAUSSIAN_MIXTURE = ScaledGaussianMixture(
    0.5, np.exp(-0), np.exp(-6)
)