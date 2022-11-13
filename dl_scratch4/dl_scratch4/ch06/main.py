from classopt import classopt, config
import pretty_errors

from dl_scratch4.common.gridworld import GridWorld, Coord
from dl_scratch4.ch06.agents.td_eval import TdAgent
from dl_scratch4.ch06.agents.sarsa import SarsaAgent
from dl_scratch4.ch06.agents.sarsa_off_policy import SarsaOffPolicyAgent


AGENTS = {
    'sarsa': SarsaAgent(),
    'off_sarsa': SarsaOffPolicyAgent(),
}


@classopt
class Args:
    agent_type: str = config(long=True, short="-a")


def main(episodes: int = 1000):
    agent_type = Args.from_args().agent_type
    assert agent_type in AGENTS.keys()
    agent = AGENTS[agent_type]

    env: GridWorld = GridWorld()

    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            if done:
                agent.update(next_state, None, None, None, None)
                break
            state = next_state

    env.render_q(agent.Q)


if __name__ == "__main__":
    main()
