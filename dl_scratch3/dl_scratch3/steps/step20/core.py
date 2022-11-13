from __future__ import annotations
from abc import ABC, abstractmethod
from weakref import ref, ReferenceType
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from dl_scratch3.steps.step20.config import Config


@dataclass
class Variable:
    data: np.ndarray
    name: str | None = None
    grad: np.ndarray | None = None
    creator: Function | None = None
    generation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            msg = f'{type(self.data)} is not supported. The type must be numpy.ndarray.'
            raise TypeError(msg)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p: str = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np._DType_co:
        return self.data.dtype

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self, retain_grad: bool = False) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Function] = []
        seen_set: set[Function] = set()

        def inner_add_func(func: Function):
            if func not in seen_set:
                funcs.append(func)
                seen_set.add(func)
                funcs.sort(key=lambda x: x.generation)
        
        inner_add_func(self.creator)
        while funcs:
            func = funcs.pop()
            gys: list[np.ndarray] = [output().grad for output in func.outputs]
            gxs: np.ndarray | tuple[np.ndarray, ...] = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            assert len(func.inputs) == len(gxs)
            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # x.grad += gx は不可。 b/c ndarrayインスタンスのコピーを生成しないため
                if x.creator is not None:
                    inner_add_func(x.creator)
            if not retain_grad:
                for y in func.outputs:
                    y().grad = None  # 微分係数を保持しない


class Function(ABC):
    def __call__(self, *inputs: Variable) -> Variable | list[Variable]:
        xs: list[np.ndarray] = [x.data for x in inputs]
        ys: np.ndarray | tuple[np.ndarray] = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs: list[Variable] = [Variable(self._as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation: int = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs: tuple[Variable, ...] = inputs
            self.outputs: list[ReferenceType[Variable]] = [ref(output) for output in outputs]
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


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


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


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


def mul(x0: Variable, x1: Variable) -> Variable:
    return Mul()(x0, x1)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


Variable.__mul__ = mul
Variable.__add__ = add
