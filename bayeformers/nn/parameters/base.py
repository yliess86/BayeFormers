# -*- coding: utf-8 -*-
"""bayeformers.nn.parameters.base

The files provides base classes and methods for creating bayesian parameters.
These parameters provides sampling capabilities and access to their
corresponding log probabilities. 
"""
from torch import Size
from torch import Tensor
from torch.nn import Module
from typing import Optional

import torch
import torch.nn as nn


def parameter(
    size: Size, dtype: Optional[torch.dtype] = torch.float32
) -> nn.Parameter:
    """parameter

    Proxy for the creation of a parameter given a size and a data type.

    Arguments:
        size (Size): size of the parameter
        dtype (Optional[torch.dtype], optional): type of the parameter's data.
            {default: torch.float32}.

    Returns:
        nn.Parameter: zeros tensor of given size returned as a parameter
    """
    return nn.Parameter(torch.zeros(size, dtype=torch.float32))


class Parameter(Module):
    """Parameter

    Base class for bayesian Parameters. A Bayesian parameter as used in
    bayeformers must provide two methods: one for sampling from the parameter,
    most of the time using the reparametrization trick allowing
    differentiation, and a second responsible for the computation of its
    log probability.
    """

    def __init__(self) -> None:
        super(Parameter, self).__init__()

    def sample(self) -> Tensor:
        raise NotImplementedError("Sample not implemented yet")
    
    def log_prob(self, input: Tensor) -> Tensor:
        raise NotImplementedError("Log_prob not implemented yet")


class NoneParameter(Parameter):
    """NoneParameter

    Proxy for None weights. This is useful for initialization bias and calling
    sample and log_prob methods without including a special conditional branch.
    """

    def __init__(self) -> None:
        super(NoneParameter, self).__init__()
        
    def sample(self) -> Tensor:
        return None

    def log_prob(self, input: Tensor) -> Tensor:
        return 0.0