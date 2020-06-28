import torch
import torch.nn

from typing import Optional

from bayeformers import utils
from bayeformers.nn.model import Model
from bayeformers.nn.parameters.initializations import Initialization, DEFAULT_UNIFORM
from bayeformers.nn.parameters.base import Parameter
from bayeformers.nn.parameters.gaussian import DEFAULT_SCALED_GAUSSIAN_MIXTURE

"""
    Entrypoint of Bayeformers, call to_bayesian on your PyTorch model to
    replace in-place the layers with bayesians layers.

    params
        model: torch.nn.Model The model to process
        initialization: bayeformers.nn.parameters.initializations.Initialization How to initialize the bayesian layers
        prior: bayeformers.nn.parameters.gaussian.Parameter The prior parameters
"""
def to_bayesian(
    model: nn.Model,
    initialization: Optional[Initialization] = DEFAULT_UNIFORM,
    prior: Optional[Parameter] = DEFAULT_SCALED_GAUSSIAN_MIXTURE,
) -> None:
    utils.replace_modules(model, initialization, prior)
    return Model(model)
