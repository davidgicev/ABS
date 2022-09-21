import math
from copy import deepcopy
from random import randint

import gym as gym
import numpy as np
import pygame as pygame

from gym import spaces


def move_to_direction(pos, direction):
    x, y = pos
    x = x + ((1 - direction) if (direction % 2 == 0) else 0)
    y = y + ((direction - 2) if (direction % 2 != 0) else 0)
    return x, y


def generate_unique(generated, range):
    x = randint(0, range[0] - 1)
    y = randint(0, range[1] - 1)

    while (x, y) in generated:
        x = randint(0, range[0] - 1)
        y = randint(0, range[1] - 1)

    return x, y


class SnakeGame(gym.Env):
    AGENT_COLORS = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    class Agent:
        def __init__(self):
            self.direction = randint(0, 3)
            self.head = (0, 0)
            self.body = [(0, 0)]
            self.dead = False

        def reset(self, x, y):
            self.direction = randint(0, 3)
            self.head = (x, y)
            self.body = [(x, y)]
            self.dead = False

        def move(self, action, ate_food):
            self.direction = (self.direction + action) % 4
            moved = move_to_direction(self.head, self.direction)
            self.head = moved
            self.body.insert(0, deepcopy(moved))
            if not ate_food:
                self.body.pop()

        def draw(self, scale, surface, image):
            for pos in self.body:
                surface.blit(image, ((pos[0]+1)*scale, (pos[1]+1)*scale))

    def __init__(self, num_agents=2, num_fruits=20, width=30, height=30):

        self.agents = [self.Agent() for _ in range(num_agents)]
        self.fruits = []

        self.num_agents = num_agents
        self.num_fruits = num_fruits
        self.width = width
        self.height = height
        self.scale = 18
        self.display_width = (width+2) * self.scale
        self.display_height = (height+2) * self.scale
        self.steps = 0
        self.max_steps = 1000

        self._display_surf = None
        self._image_surf = None
        self._fruit_surf = None

        self.reset()
        # self._pygame_init()

        self.observation_space = spaces.Box(low=0, high=4, shape=(self.num_agents, 2, 3))
        self.action_space = spaces.Tuple(
            [spaces.Discrete(3) for i in range(self.num_agents)]
        )

    def step(self, actions):
        rewards = [-0.1] * self.num_agents
        actions = [a-1 for a in actions]

        for i, agent in enumerate(self.agents):
            if agent.dead:
                continue

            preview = move_to_direction(agent.head, (agent.direction+actions[i])%4)
            head_snapshot = deepcopy(agent.head)
            ate_food = preview in self.fruits
            agent.move(actions[i], ate_food)

            closest = self.fruits[np.argmin([math.dist(f, agent.head) for f in self.fruits])]
            if math.dist(head_snapshot, closest) > math.dist(agent.head, closest):
                rewards[i] = 0.1

            self.fruits = [x for x in self.fruits if x != preview]
            if ate_food:
                rewards[i] = 10
                if len(self.fruits) == 0:
                    self.fruits = []
                    generated = []
                    for i in range(self.num_fruits):
                        pair = generate_unique(generated, (self.width, self.height))
                        self.fruits.append(pair)
                        generated.append(pair)
                continue

            x, y = agent.head
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                rewards[i] = -10
                agent.dead = True
                continue

            if agent.head in agent.body[1:]:
                rewards[i] = -10
                agent.dead = True
                continue

        for i, agent in enumerate(self.agents):
            if agent.dead:
                continue

            collides = False
            for j, snake in enumerate(self.agents):
                if i == j:
                    continue
                if agent.head in snake.body:
                    collides = True
                    break
            agent.dead = collides
            if collides:
                rewards[i] = -10
                continue

        done = sum([1 for x in self.agents if not x.dead]) == 0
        done = done or len(self.fruits) == 0
        self.steps = self.steps + 1
        if self.steps > self.max_steps:
            done = True


        return self.get_new_relative_state(), rewards, done, {}

    def reset(self):
        self.steps = 0
        generated = []
        for agent in self.agents:
            x, y = generate_unique(generated, (self.width, self.height))
            agent.reset(x, y)
            generated.append((x, y))

        self.fruits = []
        for i in range(self.num_fruits):
            pair = generate_unique(generated, (self.width, self.height))
            self.fruits.append(pair)
            generated.append(pair)

        return self.get_new_relative_state()
    # Test
    def get_new_relative_state(self):
        states = []
        for agent in self.agents:
            if self.num_agents == 1:
                other_snake_bodies = []
            else:
                other_snake_bodies = [b for b in [a.body for a in self.agents if a != agent]][0]

            closest = self.fruits[np.argmin([math.dist(f, agent.head) for f in self.fruits])]
            x, y = agent.head
            bodies_right = sum([1 for b in other_snake_bodies if 0 < b[0] - x < 3 and math.dist(b, agent.head) < 5]) > 1
            bodies_left = sum([1 for b in other_snake_bodies if 0 < x - b[0] < 3 and math.dist(b, agent.head) < 5]) > 1
            bodies_down = sum([1 for b in other_snake_bodies if 0 < b[1] - y < 3 and math.dist(b, agent.head) < 5]) > 1
            bodies_up = sum([1 for b in other_snake_bodies if 0 < y - b[1] < 3 and math.dist(b, agent.head) < 5]) > 1
            state = (
                int(closest[0] > x),
                int(closest[0] < x),
                int(closest[1] > y),
                int(closest[1] < y),
                int(x == 0),
                int(y == 0),
                int(x == self.width - 1),
                int(y == self.height - 1),
                int(bodies_left or (x - 1, y) in agent.body),
                int(bodies_up or (x, y - 1) in agent.body),
                int(bodies_right or (x + 1, y) in agent.body),
                int(bodies_down or (x, y + 1) in agent.body),
                int(agent.direction == 0),
                int(agent.direction == 1),
                int(agent.direction == 2),
                int(agent.direction == 3),
            )
            states.append(state)
        return np.array(states)

    def get_povs(self):
        povs = []
        for a, agent in enumerate(self.agents):
            vx = move_to_direction((0, 0), agent.direction)
            vy = move_to_direction((0, 0), (agent.direction-1) % 4)
            pov = np.zeros((5, 7))
            hx, hy = agent.head
            for i in range(5):
                for j in range(7):
                    x = hx + vx[0]*i + vy[0]*(3-j)
                    y = hy + vx[1]*i + vy[1]*(3-j)
                    if x < 0 or x >= self.width or y < 0 or y >= self.height:
                        pov[i][j] = -1
                    elif (x, y) in self.fruits:
                        pov[i][j] = 1
                    elif (x, y) in agent.body:
                        pov[i][j] = -1
                    else:
                        other = False
                        for other_a, other_agent in enumerate(self.agents):
                            if a == other_a:
                                continue
                            if (x, y) in other_agent.body:
                                other = True
                                break
                        pov[i][j] = -1 if other else 0

            povs.append(pov)
        return povs

    def close(self):
        pygame.quit()

    def render(self):

        self._pygame_init()

        self._draw_env()
        for f in self.fruits:
            self._pygame_draw(self._display_surf, self._fruit_surf, f)

        for i, a in enumerate(self.agents):
            if a.dead:
                continue
            a.draw(self.scale, self._display_surf, self._agent_surfs[i])

        pygame.display.flip()

    def _draw_env(self):
        self._display_surf.fill((0, 110, 110))

        for i in range(0, self.display_width, self.scale):
            self._display_surf.blit(self._wall_surf, (0, i))
            self._display_surf.blit(self._wall_surf, (self.display_width - self.scale, i))

        for i in range(0, self.display_width, self.scale):
            self._display_surf.blit(self._wall_surf, (i, 0))
            self._display_surf.blit(self._wall_surf, (i, self.display_width - self.scale))

    def _pygame_draw(self, surface, image, pos):
        surface.blit(image, ((pos[0]+1)*self.scale, (pos[1]+1)*self.scale))

    def _pygame_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.display_width, self.display_height), pygame.HWSURFACE)
        self._agent_surfs = []

        for i, p in enumerate(self.agents):
            image_surf = pygame.Surface([self.scale, self.scale])
            image_surf.fill(self.AGENT_COLORS[i % len(self.AGENT_COLORS)])
            self._agent_surfs.append(image_surf)

        self._fruit_surf = pygame.Surface([self.scale, self.scale])
        self._fruit_surf.fill((255, 0, 0))

        self._wall_surf = pygame.Surface([self.scale, self.scale])
        self._wall_surf.fill((255, 255, 255))
