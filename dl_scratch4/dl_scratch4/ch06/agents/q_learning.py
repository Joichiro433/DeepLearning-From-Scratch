from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from dl_scratch4.common.gridworld import GridWorld, Coord
from dl_scratch4.common.utils import greedy_probs


class QLearningAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.alpha: float = 0.8
        self.epsilon: float = 0.1
        self.action_size: int = 4

        random_actions: Dict[int, float] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi: Dict[Coord, Dict[int, float]] = defaultdict(lambda: random_actions)
        self.b: Dict[Coord, Dict[int, float]] = defaultdict(lambda: random_actions)
        self.Q: Dict[Tuple[Coord, int], float] = defaultdict(lambda: 0)

    def get_action(self, state: Coord) -> int:
        action_probs: Dict[int, float] = self.b[state]
        actions: List[int] = list(action_probs.keys())
        probs: List[float] = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state: Coord, action: int, reward: float, next_state: Coord, done: bool) -> None:
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = QLearningAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_q(agent.Q)