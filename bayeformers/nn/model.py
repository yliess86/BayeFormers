# -*- coding: utf-8 -*-
"""bayeformers.nn.model

The files provides bassis for building a Bayesian Model.
"""
from torch import Tensor
from torch.nn import Module
from typing import Any
from typing import Iterator
from typing import Optional

import torch
import warnings


def is_module_bayesian(module: Module) -> bool:
    """Is Module Bayesian

    Arguments:
        module (Module): module to be checked

    Returns:
        bool: is the module bayesian or not (does it provides access to
            log_prior and log_variational_posterior)
    """
    log_prior = hasattr(module, "log_prior")
    log_variational_posterior = hasattr(module, "log_variational_posterior")
    return log_prior and log_variational_posterior


class Model(Module):
    """Model

    Wrapper for building a Bayesian Module. Provides methods to gather
    log_prior and log_variational_posterior from child bayesian modules.

    Attributes:
        model (Optional[nn.Module]): base module, useful when creating the
            bayesian module out of an exisiting frequentist one {default: None}
    """

    def __init__(self, model: Optional[Module] = None) -> None:
        """Initialization

        Arguments:
            model (Optional[nn.Module]): base module, useful when creating the
                bayesian module out of an exisiting frequentist one
                {default: None}
        """
        super(Model, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs) -> Any:
        """Forward"""
        if self.model is not None:
            return self.model.forward(*args, **kwargs)
        raise NotImplementedError("Forward pass not implemented yet")

    @property
    def bayesian_children(self) -> Iterator[Module]:
        """Bayesian Children

        Yields:
            Iterator[nn.Module]: all beysian children modules
        """
        children = filter(is_module_bayesian, self.modules())
        children = [c for c in children if c != self]
        return children

    def log_prior(self) -> Tensor:
        children = list(self.bayesian_children)
        if not len(children):
            warnings.warn("No Bayesian Child is present in this model")

        value = 0.0
        for child in children:
            value += child.log_prior
        return value


    def log_variational_posterior(self) -> Tensor:
        children = list(self.bayesian_children)
        if not len(children):
            warnings.warn("No Bayesian Child is present in this model")

        value = 0.0
        for child in children:
            value += child.log_variational_posterior
        return value