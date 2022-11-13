from typing import Any, Callable

import numpy as np
from rich import print
import pretty_errors
from dl_scratch3.steps.step01 import Variable
from dl_scratch3.steps.step02 import Square
from dl_scratch3.steps.step03 import Exp


def numerical_diff(f: Callable[..., Variable], x: Variable, eps: float = 1e-4) -> Any:
    x0, x1 = Variable(x.data - eps), Variable(x.data + eps)
    y0, y1 = f(x0), f(x1)
    return (y1.data - y0.data) / (2 * eps)


if __name__ == '__main__':
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

    def func(x):
        f1 = Square()
        f2 = Exp()
        f3 = Square()
        return f3(f2(f1(x)))

    x = Variable(np.array(0.5))
    dy = numerical_diff(func, x)
    print(dy)



