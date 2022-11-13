import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step19.functions import *
from dl_scratch3.steps.step19.config import no_grid


if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array([
        [1, 2, 3],
        [4, 5, 6]
    ]))
    print(x0)
    print(x1)
