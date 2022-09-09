import time

import gym
import numpy as np

from custom_snake_env import SnakeGame
from deep_q_learning import DQN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()

    model.add(Dense(16, activation='relu', input_dim=state_space_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))

    model.compile(Adam(lr=learning_rate), loss=MeanSquaredError())
    model.summary()

    return model


if __name__ == '__main__':
    env = SnakeGame(num_fruits=1)
    # env._max_episode_steps = 1000
    # env.reset()

    state_space_shape = 12  # env.observation_space.shape[0]
    print(env.observation_space)
    num_actions = 3
    print(env.action_space)

    num_episodes = 100
    learning_rate = 0.1
    discount_factor = 0.99
    batch_size = 128
    memory_size = 1024
    epsilon = 0.2
    epsilon_decay = 0.025
    min_epsilon = 0.1

    model = build_model(state_space_shape, num_actions, learning_rate)
    target_model = build_model(state_space_shape, num_actions, learning_rate)

    agent = DQN(state_space_shape, num_actions, model, target_model,
                learning_rate, discount_factor, batch_size, memory_size)

    # agent.load("snake_dqn_food_dir_vector", 89)

    temp_epsilon = epsilon

    for episode in range(num_episodes):
        print("Episode", episode)
        state = env.reset()
        done = False
        sum_rewards = 0
        steps = 0
        start_time = time.time()
        while not done:
            actions = [agent.get_action(np.reshape(s, -1), temp_epsilon) for s in state]
            # actions = [0]*env.num_agents
            temp = state.copy()
            new_state, rewards, done, _ = env.step(actions)
            for a in range(env.num_agents):
                # t_state = np.reshape(new_state[a], -1)
                agent.update_memory(temp[a], actions[a], rewards[a], new_state[a], done)
            state = new_state
            sum_rewards = sum_rewards + sum(rewards)
            steps = steps + 1
            if episode % 10 == 0 and episode > 50:
                env.render()
                time.sleep(0.16)
        print("rewards", sum_rewards)
        print("steps", steps)
        print("steps/s", steps/(time.time() - start_time))
        print("Training model")
        agent.train()
        if episode % 3 == 0:
            agent.update_target_model()
        if episode % 10 == 0:
            agent.save("snake_dqn_food_dir_vector", episode - 1)

        if temp_epsilon > min_epsilon:
            temp_epsilon = temp_epsilon - epsilon_decay


    print("Testing")

    num_iterations = 50  # ili 100

    sum_reward = 0
    sum_steps = 0
    for _ in range(num_iterations):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state, 0)
            state, reward, done, _ = env.step(action)
            sum_steps = sum_steps + 1
            sum_reward = sum_reward + reward

    print("  On 50 test iterations")
    print("    avg steps", sum_steps / num_iterations)
    print("    avg reward", sum_reward / num_iterations)

    done = False
    state = env.reset()
    # env.render()
    while not done:
        action = agent.get_action(state, 0)
        state, reward, done, _ = env.step(action)
        # env.render()
