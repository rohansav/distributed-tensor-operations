from __future__ import annotations
import numpy as np


class Tensor:
  def __init__(self, data: np.array, requires_grad: bool = False) -> None:
    self.data = data
    self.requires_grad = requires_grad

  def __add__(self, other: Tensor) -> Tensor:
    return Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

  def __mul__(self, other: Tensor) -> Tensor:
    return Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)


