from torch import Tensor
from torch.nn import Module
from typing import Iterator

import torch


def is_module_bayesian(module: Module) -> bool:
    log_prior = hasattr(modules, "log_prior")
    log_variational_posterior = hasattr(modules, "log_variational_posterior")
    return log_prior and log_variational_posterior


class Model(Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

    def forward(self):
        raise NotImplemented("Forward pass not implemented yet")

    @property
    def bayesian_children(self) -> Iterator[Module]:
        return filter(is_module_bayesian, self.children())

    def log_prior(self) -> Tensor:
        return torch.sum([
            child.log_prior for child in self.bayesian_children
        ])

    def log_variational_posterior(self) -> Tensor:
        return torch.sum([
            child.log_variational_posterior for child in self.bayesian_children
        ])