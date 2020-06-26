from torch import Tensor

import numpy as np
import torch


def reparametrization_trick(eps: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    return mu + eps * sigma


def gaussian_log_prob(input: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    return (
        - np.log(np.sqrt(2 * np.pi))
        - torch.log(sigma)
        - ((input - mu) ** 2) / (2 * sigma ** 2)
    ).sum()


def scaled_gaussian_mixture_log_prob(
    input: Tensor, log_prob1: Tensor, log_prob2: Tensor, pi: float
) -> Tensor:
    prob1, prob2 = torch.exp(log_prob1), torch.exp(log_prob2)
    return torch.log(pi * prob1 + (1.0 - pi) * prob2).sum()