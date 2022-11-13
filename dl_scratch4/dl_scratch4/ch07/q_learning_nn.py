from typing import List, Dict, Tuple

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from dl_scratch4.common.gridworld import GridWorld, Coord
from rich import print


sns.set_style('whitegrid')


def one_hot(state: Coord) -> NDArray[Shape['Batch, 12'], Float]:
    HEIGHT, WIDTH = 3, 4
    vec: NDArray[Shape['12'], Float] = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx: int = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)  # hidden_size
        self.l2 = L.Linear(4)  # action_size

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma: float = 0.9
        self.lr: float = 0.01
        self.epsilon: float = 0.1
        self.action_size: int = 4

        self.qnet: QNet = QNet()
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(
            self, 
            state_vec: NDArray[Shape['Batch, 12'], Float]) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(
            self,
            state: NDArray[Shape['Batch, 12'], Float], 
            action: int, 
            reward: float, 
            next_state: NDArray[Shape['Batch, 12'], Float], 
            done: bool) -> float:
        if done:
            next_q = np.zeros(1)  # [0.]
        else:
            next_qs: NDArray[Shape['Batch, Action'], Float] = self.qnet(next_state)
            next_q: NDArray[Shape['Batch'], Float] = next_qs.max(axis=1)
            next_q.unchain()

        target: NDArray[Shape['Batch, Action'], Float] = self.gamma * next_q + reward
        qs: NDArray[Shape['Batch, Action'], Float] = self.qnet(state)
        q: NDArray[Shape['Batch'], Float] = qs[:, action]
        loss: float = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data


env = GridWorld()
agent = QLearningAgent()

episodes = 1000
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)


plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# visualize
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)