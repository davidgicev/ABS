import pickle
import time

import numpy as np

from custom_snake_env import SnakeGame
from deep_q_learning import DQN

import matplotlib.pyplot as plt


class TestingComponent:
    def __init__(self, env: SnakeGame, dqn: DQN):
        self.snapshots = []
        self.episodes = 10
        self.env = env
        self.dqn = dqn

    def start_testing(self, render=False):
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            sum_rewards = 0
            steps = 0
            agent_dead = [False] * self.env.num_agents
            while not done:
                actions = [self.dqn.get_action(np.reshape(s, -1), 0) for s in state]
                temp = state.copy()
                new_state, rewards, done, _ = self.env.step(actions)
                for a in range(self.env.num_agents):
                    if agent_dead[a]:
                        continue
                    if rewards[a] == -10:
                        agent_dead[a] = True
                    self.dqn.update_memory(temp[a], actions[a], rewards[a], new_state[a], done)
                state = new_state
                sum_rewards = sum_rewards + sum(rewards)
                steps = steps + 1
                if render:
                    self.env.render()
                    time.sleep(0.1)
            # print("rewards", sum_rewards)
            # print("steps", steps)
            # print("steps/s", steps/(time.time() - start_time + 0.001))
            # print("Training model")
            self.snapshots.append(sum_rewards)

    def save_results(self):
        with open('results.pickle', 'wb') as handle:
            pickle.dump(self.snapshots, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        file_name = path + "/results.pickle"

        file = open(file_name, 'rb')
        results = pickle.load(file)
        file.close()
        print(results)
        self.snapshots = results

    def graph_results(self):
        episodes = range(len(self.snapshots))

        plt.plot(episodes, self.snapshots)
        plt.title('Rewards over time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()
