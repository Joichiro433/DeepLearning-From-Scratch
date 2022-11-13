from typing import List, Dict
from collections import defaultdict

from dl_scratch4.common.gridworld import GridWorld, Coord
from dl_scratch4.ch04.policy_iter import greedy_policy


def value_iter_onestep(V: Dict[Coord, float], env: GridWorld, gamma: float) -> Dict[Coord, float]:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values: List[float] = []
        for action in env.actions():
            next_state: Coord = env.next_state(state, action)
            r: float = env.reward(state, action, next_state)
            value: float = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V


def value_iter(V: Dict[Coord, float], env: GridWorld, gamma: float, threshold: float = 0.001, is_render: bool = True) -> Dict[Coord, float]:
    while True:
        if is_render:
            env.render_v(V)

        old_V: Dict[Coord, float] = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta: float = 0
        for state in V.keys():
            t: float = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
