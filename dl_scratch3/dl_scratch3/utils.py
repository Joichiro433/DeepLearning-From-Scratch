from __future__ import annotations
import os
from pathlib import Path
import subprocess

from dl_scratch3.variable import Variable
from dl_scratch3.functions.base_function import Function


VAR_COLOR = '#f63366'
FUNC_COLOR = '#33c8f6'
ARROW_COLOR = '#444444'


def plot_dot_graph(output: Variable, verbose: bool = True, to_file: Path | str = Path('graph.pdf')) -> None:
    dot_graph: str = get_dot_graph(output=output, verbose=verbose)
    temp_dir: Path = Path('~/.dezero').expanduser()
    os.makedirs(temp_dir, exist_ok=True)
    graph_path: Path = temp_dir / 'temp_graph.dot'
    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    
    if isinstance(to_file, str):
        to_file = Path(to_file)
    extension: str = to_file.suffix[1:]  # e.g. pdf
    cmd: str = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(cmd, shell=True)


def get_dot_graph(output: Variable, verbose: bool = True) -> str:
    txt: str = ''
    funcs: list[Function] = []
    seen_set: set[Function] = set()

    def inner_add_func(func: Function) -> None:
        if func not in seen_set:
            funcs.append(func)
            seen_set.add(func)
    
    inner_add_func(func=output.creator)
    txt += _dot_var(var=output, verbose=verbose)

    while funcs:
        func: Function = funcs.pop()
        txt += _dot_func(func=func)
        for x in func.inputs:
            txt += _dot_var(var=x, verbose=verbose)
            if x.creator is not None:
                inner_add_func(func=x.creator)
    return 'digraph g {\n edge [color = "#444444"];\n' + txt + '}'  



def _dot_var(var: Variable, verbose: bool = False) -> str:
    name: str = '' if var.name is None else var.name
    if verbose and var.data is not None:
        if var.name is not None:
            name += ': '
        name += f'{var.shape} {var.dtype}'
    dot_var: str = f'{id(var)} [label="{name}", color="{VAR_COLOR}", style=filled, fontcolor=white]\n'
    return dot_var


def _dot_func(func: Function) -> str:
    txt: str = f'{id(func)} [label="{func.__class__.__name__}", color="{FUNC_COLOR}", style=filled, shape=box, fontcolor=white]\n'
    for x in func.inputs:
        txt += f'{id(x)} -> {id(func)}\n'
    for y in func.outputs:
        txt += f'{id(func)} -> {id(y())}\n'
    return txt
