from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from nptyping import NDArray, Shape, Int, Float
from dl_scratch4.common.gridworld import GridWorld, Coord


class RandomAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.action_size: int = 4

        random_actions: Dict[int, float] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi: Dict[Coord, Dict[int, float]] = defaultdict(lambda: random_actions)
        self.V: Dict[Coord, float] = defaultdict(lambda: 0)
        self.cnts: Dict[Coord, int] = defaultdict(lambda: 0)
        self.memory: List[Tuple[Coord, int, float]] = []

    def get_action(self, state: Coord) -> int:
        action_probs: Dict[int, float] = self.pi[state]
        actions: List[int] = list(action_probs.keys())
        probs: List[float] = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state: Coord, action: int, reward: float) -> None:
        data: Tuple[Coord, int, float] = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def update(self) -> None:
        G: float = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]
