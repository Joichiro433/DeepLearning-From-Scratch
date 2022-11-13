import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step16.functions import *


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y)
    print(x)
