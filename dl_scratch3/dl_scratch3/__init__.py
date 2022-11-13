from dl_scratch3.functions.config import no_grid
from dl_scratch3.variable import Variable
from dl_scratch3.functions.functions import *
from dl_scratch3.utils import plot_dot_graph


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = sub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow

setup_variable()
