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
from copy import deepcopy
from typing import Optional

import torch
import torch.nn


def to_bayesian(
    model: nn.Model,
    initialization: Optional[Initialization] = DEFAULT_UNIFORM,
    prior: Optional[Parameter] = DEFAULT_SCALED_GAUSSIAN_MIXTURE,
    delta: float = None, freeze: bool = False
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
        delta (float): is the model pretrained? If it not None, the model
            weights will be used to initialize the bayesian weights and delta
            will be used for MOPED posterior init {default: None}
            pretrained loading following:
                    "Specifying Weight Priors in Bayesian Deep Neural Networks
                    with Empirical Bayes" from Krishnan et al.
            reference: https://arxiv.org/pdf/1906.05323.pdf
        freeze (bool): freeze weight's mu if delta is not None {default: False}

    Returns:
        Model: provided model as a bayesian
    """
    def replace_layers(model, init, prior, delta):
        for name, layer in model.named_children():
            if layer.__class__ in TORCH2BAYE.keys():
                params = init, prior, delta, freeze
                bayesian = TORCH2BAYE[layer.__class__]
                bayesian = bayesian.from_frequentist(layer, *params)
                setattr(model, name, bayesian)
            replace_layers(layer, init, prior, delta)

    new_model = deepcopy(model)
    replace_layers(new_model, initialization, prior, delta)
    new_model = Model(model=new_model)
    
    return new_model