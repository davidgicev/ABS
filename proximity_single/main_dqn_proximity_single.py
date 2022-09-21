import pathlib
import random
import time

import gym
import numpy as np

import deep_q_learning
from custom_snake_env import SnakeGame
from deep_q_learning import DQN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import os

from testing import TestingComponent

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()

    model.add(Dense(128, activation='relu', input_dim=state_space_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))

    model.compile(Adam(lr=learning_rate), loss=MeanSquaredError())
    model.summary()

    return model


if __name__ == '__main__':
    env = SnakeGame(num_fruits=1, width=20, height=20, num_agents=1)
    env.max_steps = 1000

    headstart = 1000

    state_space_shape = 16
    num_actions = 3

    num_episodes = 2001
    learning_rate = 0.00025
    discount_factor = 0.99
    batch_size = 128
    memory_size = 1028
    epsilon = 0.4
    epsilon_decay = 0.02
    min_epsilon = 0.05

    model = build_model(state_space_shape, num_actions, learning_rate)
    target_model = build_model(state_space_shape, num_actions, learning_rate)

    dqn = DQN(state_space_shape, num_actions, model, target_model,
              learning_rate, discount_factor, batch_size, memory_size)

    temp_epsilon = epsilon

    testing_component = TestingComponent(env, dqn)
    testing_component.load(str(pathlib.Path().resolve()))

    dqn.load("weights", headstart)

    for episode in range(headstart, num_episodes):
        print("Episode", episode)
        state = env.reset()
        done = False
        sum_rewards = 0
        steps = 0
        start_time = time.time()
        agent_dead = [False] * env.num_agents
        while not done:
            actions = [dqn.get_action(np.reshape(s, -1), temp_epsilon) for s in state]
            temp = state.copy()
            new_state, rewards, done, _ = env.step(actions)
            for a in range(env.num_agents):
                if agent_dead[a]:
                    continue
                if rewards[a] == -10:
                    agent_dead[a] = True
                dqn.update_memory(temp[a], actions[a], rewards[a], new_state[a], done)
            state = new_state
            sum_rewards = sum_rewards + sum(rewards)
            steps = steps + 1
        # print("rewards", sum_rewards)
        # print("steps", steps)
        # print("steps/s", steps/(time.time() - start_time + 0.001))
        # print("Training model")
        dqn.train()
        if episode % 3 == 0:
            dqn.update_target_model()

        if temp_epsilon > min_epsilon:
            temp_epsilon = temp_epsilon - epsilon_decay

        if episode % 20 == 0:
            testing_component.start_testing()

    dqn.save("weights", num_episodes - 1)
    testing_component.save_results()
