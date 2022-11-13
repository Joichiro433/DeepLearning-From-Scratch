import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step09.functions import *


if __name__ == '__main__':
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)
    print(y)

    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    print(y)
    y.backward()
    print(x.grad)
