from typing import List, Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


State = Any
Action = Any


def argmax(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    selected = np.random.choice(idxes)
    return selected


def greedy_probs(Q: Dict[Tuple[State, Action], float], state: State, epsilon: float = 0, action_size: int = 4) -> Dict[Action, float]:
    qs: List[float] = [Q[(state, action)] for action in range(action_size)]
    max_action: float = np.argmax(qs)  # OR argmax(qs)

    base_prob: float = epsilon / action_size
    action_probs: Dict[Action, float] = {action: base_prob for action in range(action_size)}  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


def plot_total_reward(reward_history):
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()


