from torch.nn import Parameter
from typing import Tuple


TwoParameters = Tuple[Parameter, Parameter]
Range = Tuple[float, float]


class Initialization:
    def __call__(self, mu: Parameter, rho: Parameter) -> TwoParameters:
        raise NotImplementedError("Initialization not implemented yet")


class Uniform(Initialization):
    def __init__(self, mu_range: Range, rho_range: Range) -> None:
        self.mu_range, self.rho_range = mu_range, rho_range

    def __call__(self, mu: Parameter, rho: Parameter) -> TwoParameters:
        return mu.uniform_(self.mu_range), rho.uniform_(self.rho_range)


DEFAULT_UNIFORM = Uniform((-0.2, 0.2), (-5, -4))