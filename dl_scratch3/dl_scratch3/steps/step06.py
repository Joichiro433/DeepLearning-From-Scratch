from typing import Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from rich import print
import pretty_errors


@dataclass()
class Variable:
    data: Any
    grad: Optional[Any] = None
    

class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        assert isinstance(input, Variable), 'The type input must be Variable.'
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    @abstractmethod
    def forward(self, x: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, gy: Any) -> Any:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: Any) -> Any:
        y = x ** 2
        return y

    def backward(self, gy: Any) -> Any:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: Any) -> Any:
        y = np.exp(x)
        return y

    def backward(self, gy: Any) -> Any:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


if __name__ == "__main__":
    f1 = Square()
    f2 = Exp()
    f3 = Square()

    x = Variable(np.array(0.5))
    a = f1(x)
    b = f2(a)
    y = f3(b)
    print(y.data)

    y.grad = np.array(1.0)
    b.grad = f3.backward(y.grad)
    a.grad = f2.backward(b.grad)
    x.grad = f1.backward(a.grad)
    print(x.grad)
