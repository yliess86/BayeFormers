from torch import Tensor
from torch.nn import Module
from typing import Iterator

import torch
import warnings


def is_module_bayesian(module: Module) -> bool:
    log_prior = hasattr(module, "log_prior")
    log_variational_posterior = hasattr(module, "log_variational_posterior")
    return log_prior and log_variational_posterior


"""
    Bayesian Model wrapper with log_prior calculation
"""
class Model(Module):
    def __init__(self, replaced_model: torch.nn.Module) -> None:
        super(Model, self).__init__()
        self.replaced_model = replaced_model # the model with layers replaced with bayeformers layers

    def forward(self, *args, **kwargs):
        return self.replaced_model.forward(*args, **kwargs)

    @property
    def bayesian_children(self) -> Iterator[Module]:
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