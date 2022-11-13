from __future__ import annotations

import numpy as np

from dl_scratch3.variable import Variable
from dl_scratch3.functions.base_function import Function, _as_array


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y
    
    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gy, gy


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 - x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gy, -gy


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 / x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / (x1**2))
        return gx0, gx1


class Pow(Function):
    def __init__(self, c: float) -> None:
        self.c = c
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** self.c
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** 2
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def add(x0: Variable | np.ndarray, x1: Variable | np.ndarray | float) -> Variable:
    x1: np.ndarray = _as_array(x1)
    return Add()(x0, x1)


def sub(x0: Variable | np.ndarray, x1: Variable | np.ndarray | float) -> Variable:
    x1 = _as_array(x1)
    return Sub()(x0, x1)


def rsub(x0: Variable | np.ndarray, x1: Variable | np.ndarray | float) -> Variable:
    x1 = _as_array(x1)
    return Sub()(x1, x0)


def mul(x0: Variable | np.ndarray, x1: Variable | np.ndarray | float) -> Variable:
    x1: np.ndarray = _as_array(x1)
    return Mul()(x0, x1)


def div(x0: Variable | np.ndarray, x1: Variable | np.ndarray | float) -> Variable:
    x1 = _as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable | np.ndarray, x1: Variable | np.ndarray | float) -> Variable:
    x1 = _as_array(x1)
    return Div()(x1, x0)


def pow(x: Variable | np.ndarray, c: float) -> Variable:
    return Pow(c)(x)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)
    