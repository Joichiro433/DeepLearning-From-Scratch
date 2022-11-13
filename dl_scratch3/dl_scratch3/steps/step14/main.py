import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step14.functions import *


if __name__ == '__main__':
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)

    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)
