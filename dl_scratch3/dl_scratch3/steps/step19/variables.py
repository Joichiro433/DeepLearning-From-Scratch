from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
if TYPE_CHECKING:  # To avoid a circular import
    from dl_scratch3.steps.step19.functions import Function

np.ndarray.shape

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
            