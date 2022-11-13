import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step20.core import *


if __name__ == '__main__':
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    y = a * b
    print(y)

    z = a + b
    print(z)
