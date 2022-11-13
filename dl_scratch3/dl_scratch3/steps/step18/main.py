import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step18.functions import *
from dl_scratch3.steps.step18.config import no_grid


if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()
    print(y.grad, t.grad)
    print(x0.grad, x1.grad)

    with no_grid():
        x = Variable(np.ones((100, 100, 100)))
        y = square(square(x))
