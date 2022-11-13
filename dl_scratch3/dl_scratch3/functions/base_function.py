from __future__ import annotations
from abc import ABC, abstractmethod
from weakref import ref, ReferenceType

import numpy as np
import numpy.typing as npt

from dl_scratch3.functions.config import Config
from dl_scratch3.variable import Variable


class Function(ABC):
    def __call__(self, *inputs: Variable | np.ndarray) -> Variable | list[Variable]:
        inputs: list[Variable] = [_as_variable(x) for x in inputs]
        xs: list[np.ndarray] = [x.data for x in inputs]
        ys: np.ndarray | tuple[np.ndarray] = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs: list[Variable] = [Variable(_as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation: int = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs: list[Variable] = inputs
            self.outputs: list[ReferenceType[Variable]] = [ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    @abstractmethod
    def forward(self, *xs: np.ndarray) -> np.ndarray | tuple[np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, *gys: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        raise NotImplementedError()

    def _as_variable(self, obj: Variable | np.ndarray) -> Variable:
        if isinstance(obj, Variable):
            return obj
        return Variable(obj)


def _as_array(x: npt.ArrayLike) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def _as_variable(obj: Variable | np.ndarray) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
