from typing import Any
from abc import ABC, abstractmethod

import numpy as np
from dl_scratch3.steps.step07.variables import Variable


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        if not isinstance(input, Variable):
            msg = f'{type(input)} is not supported. The type must be Variable.'
            raise TypeError(msg)
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
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
        