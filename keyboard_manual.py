import numpy as np
from pygame.locals import *
import pygame

import custom_snake_env
import time


num_agents = 1
env = custom_snake_env.SnakeGame(num_agents, width=10, height=10, num_fruits=2)
env.render()

while True:
    actions = None
    index = 0

    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        actions = np.ones(num_agents) * 0

        if keys[K_RIGHT]:
            actions[index] = 0
            break

        if keys[K_LEFT]:
            actions[index] = 2
            break

        if keys[K_UP]:
            actions[index] = 1
            break

    obs, r, done, _ = env.step(actions)
    env.render()
    print(obs)
    print(r)
    if done:
        env.reset()

    actions = None
    time.sleep(0.2)
