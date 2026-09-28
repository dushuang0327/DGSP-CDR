from torch import nn
from abc import abstractmethod
from src.types_ import *
from typing import List, Any, Tuple


class BaseAE(nn.Module):
    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor,  Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> dict:
        pass

    @abstractmethod
    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        pass