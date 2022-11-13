from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from dl_scratch4.common.gridworld import GridWorld, Coord


class QLearningAgent:
    def __init__(self):
        self.gamma: float = 0.9
        self.alpha: float = 0.8
        self.epsilon: float = 0.1
        self.action_size: int = 4
        self.Q: Dict[Tuple[Coord, int], float] = defaultdict(lambda: 0)

    def get_action(self, state: Coord) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # 探索
        else:
            qs: List[float] = [self.Q[state, a] for a in range(self.action_size)]  # 活用
            return np.argmax(qs)

    def update(self, state: Coord, action: int, reward: float, next_state: Coord, done: bool) -> None:
        if done:
            next_q_max: float = 0
        else:
            next_qs: List[float] = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max: float = max(next_qs)

        target: float = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


env = GridWorld()
agent = QLearningAgent()

episodes = 1000
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
