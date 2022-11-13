from typing import List

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns
from dl_scratch4.ch01.bandit import Bandit, Agent


sns.set_style('whitegrid')


runs: int = 200
steps: int = 1000
epsilon: float = 0.1
all_rates: NDArray[Shape['Runs, Steps'], Float] = np.zeros((runs, steps))  # (2000, 1000)

for run in range(runs):
    bandit: Bandit = Bandit()
    agent: Agent = Agent(epsilon=epsilon)
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

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()
