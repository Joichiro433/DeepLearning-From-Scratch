from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from dl_scratch4.common.gridworld import GridWorld, Coord


class TdAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.alpha: float = 0.01
        self.action_size: int = 4

        random_actions: Dict[int, float] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi: Dict[Coord, Dict[int, float]] = defaultdict(lambda: random_actions)
        self.V: Dict[Coord, float] = defaultdict(lambda: 0)

    def get_action(self, state: Coord) -> int:
        action_probs: Dict[int, float] = self.pi[state]
        actions: List[int] = list(action_probs.keys())
        probs: List[float] = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state: Coord, reward: float, next_state: Coord, done: bool) -> None:
        next_V: float = 0 if done else self.V[next_state]
        target: float = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha
