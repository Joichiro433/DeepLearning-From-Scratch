from __future__ import annotations
from typing import List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:  # To avoid a circular import
    from dl_scratch3.steps.step08.functions import Function


@dataclass
class Variable:
    data: Any
    grad: Optional[Any] = None
    creator: Optional[Function] = None

    def set_creator(self, func: Function) -> None:
        self.creator = func

    def backward(self) -> None:
        funcs: List[Function] = [self.creator]
        while funcs:
            func = funcs.pop()
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)
            