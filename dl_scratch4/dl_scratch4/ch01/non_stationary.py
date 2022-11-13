from typing import List, Dict

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

from dl_scratch4.ch01.bandit import Agent


sns.set_style('whitegrid')


class NonStatBandit:
    def __init__(self, arms: int = 10) -> None:
        self.arms: int = arms
        self.rates: NDArray[Shape['Arms'], Float] = np.random.rand(arms)

    def play(self, arm: int) -> int:
        rate: float = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)  # Add noise
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon: float, alpha: float, actions: int = 10) -> None:
        self.epsilon: float = epsilon
        self.Qs: NDArray[Shape['Actions'], Int] = np.zeros(actions)
        self.alpha: float = alpha

    def update(self, action: int, reward: int) -> None:
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))  # 探索
        return np.argmax(self.Qs)  # 活用


runs: int = 200
steps: int = 1000
epsilon: float = 0.1
alpha: float = 0.8
agent_types: List[str] = ['sample average', 'alpha const update']
results: Dict[str, NDArray[Shape['Steps'], Float]] = {}

for agent_type in agent_types:
    all_rates: NDArray[Shape['Runs, Steps'], Float] = np.zeros((runs, steps))  # (200, 1000)

    for run in range(runs):
        if agent_type == 'sample average':
            agent: Agent = Agent(epsilon=epsilon)
        else:
            agent: AlphaAgent = AlphaAgent(epsilon=epsilon, alpha=alpha)

        bandit: NonStatBandit = NonStatBandit()
        total_reward: int = 0
        rates: List[float] = []

        for step in range(steps):
            action: int = agent.get_action()
            reward: int = bandit.play(arm=action)
            agent.update(action=action, reward=reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates: NDArray[Shape['Steps'], Float] = np.average(all_rates, axis=0)
    results[agent_type] = avg_rates

# plot
plt.figure()
plt.ylabel('Average Rates')
plt.xlabel('Steps')
for key, avg_rates in results.items():
    plt.plot(avg_rates, label=key)
plt.legend()
plt.show()