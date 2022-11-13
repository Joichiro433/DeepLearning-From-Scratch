from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
if TYPE_CHECKING:  # To avoid a circular import
    from dl_scratch3.steps.step09.functions import Function


@dataclass
class Variable:
    data: np.ndarray
    grad: Optional[np.ndarray] = None
    creator: Optional[Function] = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            msg = f'{type(self.data)} is not supported. The type must be np.ndarray.'
            raise TypeError(msg)

    def set_creator(self, func: Function) -> None:
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs: List[Function] = [self.creator]
        while funcs:
            func = funcs.pop()
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)
            