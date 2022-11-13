from typing import Dict

import numpy as np
from dl_scratch4.common.gridworld import GridWorld, Coord


env: GridWorld = GridWorld()
V: Dict[Coord, float] = {}
for state in env.states():
    V[state] = np.random.randn()  # ダミーの状態価値関数
env.render_v(V)