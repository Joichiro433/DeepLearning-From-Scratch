from typing import List, Dict, Tuple, Optional

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import dl_scratch4.common.gridworld_render as render_helper


Coord = Tuple[int, int]


class GridWorld:
    def __init__(self):
        self.action_space: List[int] = [0, 1, 2, 3]
        self.action_meaning: Dict[int, str] = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map: NDArray[Shape['Hight, Width'], Float] = np.array(
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        self.goal_state: Coord = (0, 3)
        self.wall_state: Coord = (1, 1)
        self.start_state: Coord = (2, 0)
        self.agent_state: Coord = self.start_state

    @property
    def height(self) -> int:
        return len(self.reward_map)

    @property
    def width(self) -> int:
        return len(self.reward_map[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.reward_map.shape

    def actions(self) -> List[int]:
        return self.action_space

    def states(self) -> Coord:
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state: Coord, action: int) -> Coord:
        action_move_map: List[Coord] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move: Coord = action_move_map[action]
        next_state: Coord = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state: Coord, action: int, next_state: Coord) -> float:
        return self.reward_map[next_state]

    def reset(self) -> Coord:
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action: int) -> Tuple[Coord, float, bool]:
        state: Coord = self.agent_state
        next_state: Coord = self.next_state(state, action)
        reward: float = self.reward(state, action, next_state)
        done: bool = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)
