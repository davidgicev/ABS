import numpy as np
from time import sleep

import custom_snake_env
import q_learning

if __name__ == '__main__':
    num_agents = 4
    env = custom_snake_env.SnakeGame(num_agents, width=30, height=30)
    q_table = q_learning.random_q_table(-2, -2, (5, 5, 5, 5, 5, 5, 3))
    sum_rewards = 0
    for round in range(50000):
        epsilon = 0.2
        epsilon_decay = 0.005
        state = env.reset()
        done = False
        if round % 100 == 0:
            sum_rewards = 0
        while not done:
            epsilon = max(0, epsilon - epsilon_decay)
            actions = [q_learning.get_action(env, q_table, tuple(s.astype(int).flatten()), 0.3) for s in state]
            last_state = state
            state, rewards, done, _ = env.step(actions)
            for i in range(num_agents):
                new_q = q_learning.calculate_new_q_value(q_table, tuple(last_state[i].astype(int).flatten()),
                                                         tuple(state[i].astype(int).flatten()), actions[i],
                                                         rewards[i], epsilon, 0.9)
                q_table[tuple(last_state[i].astype(int).flatten()) + (actions[i],)] = new_q
            sum_rewards = sum_rewards + sum(rewards)
            if round % 599 == 0:
                env.render()
                sleep(0.05)
        if round % 100 == 0:
            vkupno = 5*5*5*5*5*5*3
            populirani = 0
            print("Runda broj "+str(round), sum_rewards)
            for idx, value in np.ndenumerate(q_table):
                if value != -2:
                    populirani = populirani + 1
            print("Populirani %", populirani / vkupno * 100)
