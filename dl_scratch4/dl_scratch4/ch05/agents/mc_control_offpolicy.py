from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from dl_scratch4.common.gridworld import GridWorld, Coord
from dl_scratch4.common.utils import greedy_probs


class McOffPolicyAgent:
    def __init__(self):
        self.gamma: float = 0.9
        self.epsilon: float = 0.1
        self.alpha: float = 0.2
        self.action_size: int = 4

        random_actions: Dict[int, float] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi: Dict[Coord, Dict[int, float]] = defaultdict(lambda: random_actions)
        self.b: Dict[Coord, Dict[int, float]] = defaultdict(lambda: random_actions)
        self.Q: Dict[Tuple[Coord, int], float] = defaultdict(lambda: 0)
        self.memory: List[Tuple[Coord, int, float]] = []

    def get_action(self, state: Coord) -> int:
        action_probs: Dict[int, float] = self.b[state]
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
        rho: float = 1
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * rho * G + reward
            key: Tuple[Coord, int] = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha
            rho *= self.pi[state][action] / self.b[state][action]

            self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
            self.b[state] = greedy_probs(self.Q, state, epsilon=self.epsilon)
