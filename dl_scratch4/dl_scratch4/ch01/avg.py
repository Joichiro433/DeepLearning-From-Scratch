from typing import List

import numpy as np
from rich import print


# naive implementation
np.random.seed(0)
rewards: List[float] = []

for n in range(1, 11):
    reward: float = np.random.rand()
    rewards.append(reward)
    Q: float = sum(rewards) / n
    print(Q)

print('---')

# incremental implementation
np.random.seed(0)
Q = 0

for n in range(1, 11):
    reward = np.random.rand()
    # Q = Q + (reward - Q) / n
    Q += (reward - Q) / n
    print(Q)