from typing import List, Dict
from collections import defaultdict

from dl_scratch4.common.gridworld import GridWorld, Coord


def eval_onestep(pi: Dict[Coord, Dict[int, float]], V: Dict[Coord, float], env: GridWorld, gamma: float = 0.9) -> Dict[Coord, float]:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs: float = pi[state]
        new_V: float = 0
        for action, action_prob in action_probs.items():
            next_state: Coord = env.next_state(state, action)
            r: float = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(pi: Dict[Coord, Dict[int, float]], V: Dict[Coord, float], env: GridWorld, gamma: float, threshold: float = 0.001) -> Dict[Coord, float]:
    while True:
        old_V: Dict[Coord, float] = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta: float = 0
        for state in V.keys():
            t: float = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    env: GridWorld = GridWorld()
    gamma: float = 0.9

    pi: Dict[Coord, Dict[int, float]] = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V: Dict[Coord, float] = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
