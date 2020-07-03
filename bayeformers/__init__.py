# -*- coding: utf-8 -*-
"""bayeformers.__init__

The files provides the to_bayesian conversion method.
"""
from bayeformers.nn import TORCH2BAYE
from bayeformers.nn.model import Model
from bayeformers.nn.parameters.initializations import DEFAULT_UNIFORM
from bayeformers.nn.parameters.initializations import Initialization
from bayeformers.nn.parameters.base import Parameter
from bayeformers.nn.parameters.gaussian import DEFAULT_SCALED_GAUSSIAN_MIXTURE
from typing import Optional

import torch
import torch.nn


def to_bayesian(
    model: nn.Model,
    initialization: Optional[Initialization] = DEFAULT_UNIFORM,
    prior: Optional[Parameter] = DEFAULT_SCALED_GAUSSIAN_MIXTURE
) -> Model:
    """To Bayesian
    
    Entrypoint of Bayeformers, call to_bayesian on a PyTorch model to
    replace in-place the available swappable layers with their bayesian
    equivalent.

    Arguments:
        model (nn.Model): model to process
        
    Keyword Arguments:
        initialization (Optional[Initialization]): initialization callback
            for the bayesian layers
        prior (Optional[Parameter]): the prior parameters

    Returns:
        Model: provided model as a bayesian
    """
    def replace_modules(model, init, prior):
        for name, module in model.named_children():
            if module.__class__ in TORCH2BAYE.keys():
                bayesian = TORCH2BAYE[module.__class__]
                bayesian = bayesian.from_frequentist(module, init, prior)
                setattr(model, name, bayesian)
            replace_modules(module, init, prior)
        
    replace_modules(model, initialization, prior)
    return Model(model=model)