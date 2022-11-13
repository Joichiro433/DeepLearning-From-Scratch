from classopt import classopt, config
import pretty_errors

from dl_scratch4.ch05.agents.mc_eval import RandomAgent
from dl_scratch4.ch05.agents.mc_control import McAgent
from dl_scratch4.ch05.agents.mc_control_offpolicy import McOffPolicyAgent
from dl_scratch4.common.gridworld import GridWorld, Coord


AGENTS = {
    'random': RandomAgent(),
    'monte': McAgent(),
    'off_monte': McOffPolicyAgent()
}


@classopt
class Args:
    agent_type: str = config(long=True, short="-a")


def main(episodes: int = 10000):
    agent_type = Args.from_args().agent_type
    assert agent_type in AGENTS.keys()
    agent = AGENTS[agent_type]

    env: GridWorld = GridWorld()

    for episode in range(episodes):
        state: Coord = env.reset()
        agent.reset()
        while True:
            action: int = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            if done:
                agent.update()
                break
            state = next_state

    if agent_type == 'random':
        env.rander_v(agent.V)
    else:
        env.render_q(agent.Q)


if __name__ == '__main__':
    main()
