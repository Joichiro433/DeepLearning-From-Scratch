import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step13.functions import *


if __name__ == '__main__':
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(square(x0), square(x1))
    print(y)
    y.backward()
    print(x0)
    print(x1)
    