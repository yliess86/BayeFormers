# -*- coding: utf-8 -*-
"""bayeformers.nn.parameters.initializations

The files provides classes and methods for initialization bayesian parameters. 
"""
from torch.nn import Parameter
from typing import Tuple


TwoParameters = Tuple[Parameter, Parameter]
Range = Tuple[float, float]


class Initialization:
    """Initialization

    Initialization Callback for bayesian parameters.
    Must initialize mu and rho.
    """

    def __call__(self, mu: Parameter, rho: Parameter) -> TwoParameters:
        raise NotImplementedError("Initialization not implemented yet")


class Uniform(Initialization):
    """Uniform Initialization

    Attributes:
        mu_range (Range): bounds for the uniform initilaization of mu
        rho_range (Range): bounds for the uniform initilaization of rho
    """

    def __init__(self, mu_range: Range, rho_range: Range) -> None:
        """Initialization

        Arguments:
            mu_range (Range): bounds for the uniform initilaization of mu
            rho_range (Range): bounds for the uniform initilaization of rho
        """
        super(Uniform, self).__init__()
        self.mu_range, self.rho_range = mu_range, rho_range

    def __call__(self, mu: Parameter, rho: Parameter) -> TwoParameters:
        """Call

        Arguments:
            mu (nn.Parameter): mu parameter to be initialized
            rho (nn.Parameter): rho parameter to be initialized

        Returns:
            (nn.Parameter): mu initialized
            (nn.Parameter): rho initialized
        """
        mu.data = mu.data.uniform_(*self.mu_range)
        rho.data = rho.data.uniform_(*self.rho_range)
        return mu, rho


"""Default Uniform intialization"""
DEFAULT_UNIFORM = Uniform((-0.2, 0.2), (-5, -4))