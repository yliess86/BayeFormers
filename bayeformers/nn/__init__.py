# -*- coding: utf-8 -*-
"""bayeformers.nn.__init__"""
from bayeformers.nn.layers.linear import Linear


from bayeformers.nn.model import Model


from bayeformers.nn.parameters.base import NoneParameter
from bayeformers.nn.parameters.base import Parameter

from bayeformers.nn.parameters.gaussian import DEFAULT_SCALED_GAUSSIAN_MIXTURE
from bayeformers.nn.parameters.gaussian import Gaussian
from bayeformers.nn.parameters.gaussian import ScaledGaussianMixture

from bayeformers.nn.parameters.initializations import DEFAULT_UNIFORM
from bayeformers.nn.parameters.initializations import Initialization
from bayeformers.nn.parameters.initializations import Uniform


import torch.nn as nn


"""Available Bayesian Layers"""
TORCH2BAYE = { nn.Linear: Linear }