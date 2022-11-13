from typing import List, Dict, Any
from collections import defaultdict

from dl_scratch4.common.gridworld import GridWorld, Coord
from dl_scratch4.ch04.policy_eval import policy_eval


def argmax(d: Dict[Any, float]) -> Any:
    """valueが最大となるkeyを返却する"""
    max_value: float = max(d.values())
    max_key: Any = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V: Dict[Coord, float], env: GridWorld, gamma: float) -> Dict[Coord, Dict[int, float]]:
    pi: Dict[Coord, Dict[int, float]] = {}

    for state in env.states():
        action_values: Dict[int, float] = {}

        for action in env.actions():
            next_state: Coord = env.next_state(state, action)
            r: float = env.reward(state, action, next_state)
            value: float = r + gamma * V[next_state]
            action_values[action] = value

        max_action: int = argmax(action_values)
        action_probs: Dict[int, float] = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def policy_iter(env: GridWorld, gamma: float, threshold: float = 0.001, is_render: bool = True) -> Dict[Coord, Dict[int, float]]:
    pi: Dict[Coord, Dict[int, float]] = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V: Dict[Coord, float] = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi: Dict[Coord, Dict[int, float]] = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break
        pi = new_pi

    return pi


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma)
