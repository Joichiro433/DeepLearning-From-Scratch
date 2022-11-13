import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step17.functions import *


if __name__ == '__main__':
    x = Variable(np.random.randn(10000))
    y = square(square(x))
