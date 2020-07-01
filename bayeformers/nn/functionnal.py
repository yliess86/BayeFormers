# -*- coding: utf-8 -*-
"""bayeformers.nn.functionnal

The files provides functionnal utilities for the nn module ofr bayeformers.
"""
from torch import Tensor

import numpy as np
import torch


def reparametrization_trick(eps: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Reparametrization trick

    eps ~ N(0, 1)
    W = mu + eps * std

    Arguments:
        eps (Tensor): epsilon values sampled from normal distribution
        mu (Tensor): mu values
        sigma (Tensor): sigma value

    Returns:
        Tensor: raparametrized differentiable sample
    """
    return mu + eps * sigma


def gaussian_log_prob(input: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Gaussian Log Probability

    Arguments:
        input (Tensor): sampled weights
        mu (Tensor): mu values
        sigma (Tensor): sigma values

    Returns:
        Tensor: gaussian log probability
    """
    return (
        - np.log(np.sqrt(2 * np.pi))
        - torch.log(sigma)
        - ((input - mu) ** 2) / (2 * sigma ** 2)
    ).sum()


def scaled_gaussian_mixture_log_prob(
    input: Tensor, log_prob1: Tensor, log_prob2: Tensor, pi: float
) -> Tensor:
    """Scaled Gaussian Mixture Log Probability

    Arguments:
        input (Tensor): sampled weights
        log_prob1 (Tensor): log probability of the first gaussian
        log_prob2 (Tensor): log probability of the second gaussian
        pi (float): mixing factor of the two gaussian

    Returns:
        Tensor: scaled gaussian mixture log probability
    """
    prob1, prob2 = torch.exp(log_prob1), torch.exp(log_prob2)
    return torch.log(pi * prob1 + (1.0 - pi) * prob2).sum()