import pretty_errors
import numpy as np
from rich import print
from dl_scratch3.steps.step11.functions import *


if __name__ == '__main__':
    xs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    ys = f(xs)
    print(ys)