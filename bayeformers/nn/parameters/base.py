from torch import Size
from torch import Tensor
from torch.nn import Module

import torch
import torch.nn as nn


def parameter(size: Size, dtype: torch.dtype = torch.float32) -> nn.Parameter:
    return nn.Parameter(torch.zeros(size, dtype=torch.float32))


class Parameter(Module):
    def __init__(self) -> None:
        super(Parameter, self).__init__()

    def sample(self) -> Tensor:
        raise NotImplementedError("Sample not implemented yet")
    
    def log_prob(self, input: Tensor) -> Tensor:
        raise NotImplementedError("Log_prob not implemented yet")


class NoneParameter(Parameter):
    def __init__(self) -> None:
        super(NoneParameter, self).__init__()
        
    def sample(self) -> Tensor:
        return None

    def log_prob(self, input: Tensor) -> Tensor:
        return 0.0