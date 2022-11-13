from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from dl_scratch3.steps.step14.variables import Variable


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


class Function(ABC):
    def __call__(self, *inputs: Variable) -> Variable | list[Variable]:
        xs: list[np.ndarray] = [x.data for x in inputs]
        ys: np.ndarray | tuple[np.ndarray] = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs: list[Variable] = [Variable(self._as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    @abstractmethod
    def forward(self, *xs: np.ndarray) -> np.ndarray | tuple[np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, *gys: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        raise NotImplementedError()

    def _as_array(self, x: npt.ArrayLike) -> np.ndarray:
        if np.isscalar(x):
            return np.array(x)
        return x


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y
    
    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gy, gy


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
        