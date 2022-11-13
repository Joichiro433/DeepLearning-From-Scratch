import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step12.functions import *


if __name__ == '__main__':
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y)
    