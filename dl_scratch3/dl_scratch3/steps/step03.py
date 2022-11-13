from typing import Any

import numpy as np
from rich import print
import pretty_errors
from dl_scratch3.steps.step01 import Variable
from dl_scratch3.steps.step02 import Function, Square


class Exp(Function):
    def forward(self, x: Any) -> Any:
        return np.exp(x)


if __name__ == '__main__':
    f1 = Square()
    f2 = Exp()
    f3 = Square()

    x = Variable(np.array(0.5))
    a = f1(x)
    b = f2(a)
    y = f3(b)
    print(y)
