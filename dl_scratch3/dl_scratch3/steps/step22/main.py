import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step22.core import *


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    y = x + np.array(3.0)
    print(y)

    x = Variable(np.array(2.0))
    y = x + 3.0
    print(y)

    x = Variable(np.array(2.0))
    y = 3.0 + x
    print(y)

    x = Variable(np.array(2.0))
    y = np.array(3.0) + x
    print(y)

    x = Variable(np.array(2.0))
    y = np.array([3.0]) + x
    print(y)

    x = Variable(np.array(2.0))
    y = x ** 3.0
    print(y)

    x = Variable(np.array(2.0))
    y = x / 3.0
    print(y)

    x = Variable(np.array(2.0))
    y = x - 3.0
    print(y)
