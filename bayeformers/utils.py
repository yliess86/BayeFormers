import torch
import torch.nn as nn

import inspect

from bayeformers.nn.parameters.base import Parameter
from bayeformers.nn.parameters.gaussian import DEFAULT_SCALED_GAUSSIAN_MIXTURE
from bayeformers.nn.parameters.initializations import Initialization
from bayeformers.nn import AVAILABLE_LAYERS
from bayeformers.nn.model import Model
from bayeformers.nn.layers.linear import Linear

"""
    Look for the corresponding class type in the available layers
"""
def _get_bayesian_class_correspondance(module: type) -> type:
    for layer_class in AVAILABLE_LAYERS:
        if layer_class.__name__ == module.__name__:
            return layer_class
    return None


def has_candidate_replacement(module: type) -> bool:
    return _get_bayesian_class_correspondance(module) != None


"""
    Takes a toch module and return the corresponding, initialized, bayformers Model
"""
def get_bayesian_replacement_instance(
    module: nn.modules.Module, initialization: Initialization, prior: Parameter
) -> Model:
    bayesian_class: type = _get_bayesian_class_correspondance(type(module))
    if not bayesian_class:
        raise Exception(
            f"Layer {str(module)} has no Bayesian correspondance in Bayeformers"
        )
    return bayesian_class.from_frequentist(module, initialization, prior)


"""
    Recursively look for modules that can be replaced in the model and swap 
    them with the bayeformers corresponding bayesian module
"""
def replace_modules(
    net: nn.Module, initalization: Initialization, prior: Parameter
) -> None:
    for n, c in net.named_children():
        if has_candidate_replacement(type(c)):
            setattr(net, n, get_bayesian_replacement_instance(c, initalization, prior))
        replace_modules(c, initalization, prior)

