from typing import List, Tuple, Iterable
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from dl_scratch3.steps.step11.variables import Variable


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


class Function(ABC):
    def __call__(self, inputs: Iterable[Variable]) -> List[Variable]:
        # if not isinstance(input, Variable):
        #     msg = f'{type(input)} is not supported. The type must be Variable.'
        #     raise TypeError(msg)
        xs: List[np.ndarray] = [x.data for x in inputs]
        ys: List[np.ndarray] = self.forward(xs)
        outputs: List[Variable] = [Variable(self._as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _as_array(self, x: npt.ArrayLike) -> np.ndarray:
        if np.isscalar(x):
            return np.array(x)
        return x


class Add(Function):
    def forward(self, xs: Iterable[np.ndarray]) -> Tuple[np.ndarray]:
        x0, x1 = xs
        y = x0 + x1
        return (y,)
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        return super().backward(gy)


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** 2
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
        