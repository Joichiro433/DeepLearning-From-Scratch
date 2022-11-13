from typing import Any
from dataclasses import dataclass

import numpy as np
from rich import print


@dataclass(frozen=True)
class Variable:
    data: Any


if __name__ == '__main__':
    d = np.array(1.0)
    x = Variable(d)
    print(x.data)
    print(x)
    