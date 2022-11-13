from typing import List

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print


sns.set_style('whitegrid')


class Bandit:
    def __init__(self, arms: int = 10) -> None:
        self.rates: NDArray[Shape['Arms'], Float] = np.random.rand(arms)

    def play(self, arm: int) -> int:
        rate: float = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon: float, action_size: int = 10) -> None:
        self.epsilon: float = epsilon
        self.Qs: NDArray[Shape['Actions'], Float] = np.zeros(action_size)
        self.ns: NDArray[Shape['Actions'], Float] = np.zeros(action_size)

    def update(self, action: int, reward: int) -> None:
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))  # 探索
        return np.argmax(self.Qs)  # 活用


if __name__ == '__main__':
    steps: int = 1000
    epsilon: float = 0.1

    bandit: Bandit = Bandit()
    agent: Agent = Agent(epsilon)
    total_reward: int = 0
    total_rewards: List[int] = []
    rates: List[float] = []

    for step in range(steps):
        action: int = agent.get_action()
        reward: int = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print(total_reward)

    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()
