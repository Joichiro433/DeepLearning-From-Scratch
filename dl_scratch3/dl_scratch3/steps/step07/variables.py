from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:  # To avoid a circular import
    from dl_scratch3.steps.step07.functions import Function


@dataclass
class Variable:
    data: Any
    grad: Optional[Any] = None
    creator: Optional[Function] = None

    def set_creator(self, func: Function) -> None:
        self.creator = func

    def backward(self) -> None:
        func = self.creator
        if func is not None:
            x = func.input
            x.grad = func.backward(self.grad)
            x.backward()
            