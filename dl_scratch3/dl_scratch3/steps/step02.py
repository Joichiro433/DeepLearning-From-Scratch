from typing import Any
from abc import ABC

import numpy as np
from rich import print
from dl_scratch3.steps.step01 import Variable


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        assert isinstance(input, Variable), 'The type input must be Variable.'
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: Any) -> Any:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: Any) -> Any:
        return x ** 2


if __name__ == '__main__':
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)
