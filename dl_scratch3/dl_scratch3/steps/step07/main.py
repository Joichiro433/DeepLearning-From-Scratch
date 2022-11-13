import numpy as np
from rich import print
from dl_scratch3.steps.step07.functions import *


if __name__ == '__main__':
    F1 = Square()
    F2 = Exp()
    F3 = Square()

    x = Variable(np.array(0.5))
    a = F1(x)
    b = F2(a)
    y = F3(b)

    assert y.creator == F3
    assert y.creator.input == b
    assert y.creator.input.creator == F2
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == F1
    assert y.creator.input.creator.input.creator.input == x

    # y.grad = np.array(1.0)
    # F3 = y.creator
    # b = F3.input
    # b.grad = F3.backward(y.grad)
    # F2 = b.creator
    # a = F2.input
    # a.grad = F2.backward(b.grad)
    # F1 = a.creator
    # x = F1.input
    # x.grad = F1.backward(a.grad)
    # print(x.grad)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
