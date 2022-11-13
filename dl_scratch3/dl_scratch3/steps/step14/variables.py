from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
if TYPE_CHECKING:  # To avoid a circular import
    from dl_scratch3.steps.step14.functions import Function


@dataclass
class Variable:
    data: np.ndarray
    grad: np.ndarray | None = None
    creator: Function | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            msg = f'{type(self.data)} is not supported. The type must be np.ndarray.'
            raise TypeError(msg)

    def set_creator(self, func: Function) -> None:
        self.creator = func

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs: list[Function] = [self.creator]
        while funcs:
            func = funcs.pop()
            gys: list[np.ndarray] = [output.grad for output in func.outputs]
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
                    funcs.append(x.creator)
            