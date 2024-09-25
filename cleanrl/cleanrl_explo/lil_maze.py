import numpy as np
import pygame
import math
import os, imageio
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from gymnasium import spaces



class LilMaze(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, render_mode = None):
        super(LilMaze, self).__init__()

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # The size of a step
        self.step_size = 0.01

        # The maximum number of steps
        self.max_steps = 200

        # Define the initial position of the agent
        self.initial_agent_position = np.array([0.25, 0.25])

        # Define the goal position
        self.goal_position = np.array([0.25, 0.75])

        # Wall positions
        self.wall_positions = [
            [(0.0,0.5),(0.5, 0.5)],
        ]

        self.world = np.zeros((1000, 1000, 3), dtype=np.uint8)
        self.world.fill(255)
        
        self.draw(self.world, self.goal_position, (0, 255, 0))
        self.world_copy = self.world.copy()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def draw(self, world, position, color):
        pos = (int(position[0] * 1000), int(position[1] * 1000))
        world[pos[1]-2:pos[1]+2, pos[0]-2:pos[0]+2] = color


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.num_steps = 0
        self.world = self.world_copy.copy()

        # Reset the agent's position to the initial position
        self.agent_position = self.initial_agent_position
        self.draw(self.world, self.initial_agent_position, (255, 0, 0))
        self.draw(self.world_copy, self.initial_agent_position, (0, 0, 255))

        infos = self._get_info()
        return self.agent_position, infos
    
    def _get_info(self):
        return {}

    def step(self, action):
        self.num_steps += 1

        action = np.clip(action, -1, 1)
        new_position = self.agent_position + action * self.step_size

        # only made for 1 wall
        cond1 = new_position[1] >= 0.5 and self.agent_position[1] < 0.5
        cond2 = new_position[1] < 0.5 and self.agent_position[1] >= 0.5

        if cond1 and self.agent_position[0] + (new_position[0] - self.agent_position[0])/(new_position[1] - self.agent_position[1]) * (0.5 - self.agent_position[1]) < 0.5 : 
            new_position = [self.agent_position[0] + (new_position[0] - self.agent_position[0])/(new_position[1] - self.agent_position[1]) * (0.5 - self.agent_position[1]), 0.5 - 0.001]
        if cond2 and self.agent_position[0] + (new_position[0] - self.agent_position[0])/(new_position[1] - self.agent_position[1]) * (0.5 - self.agent_position[1]) < 0.5 : 
            new_position = [self.agent_position[0] + (new_position[0] - self.agent_position[0])/(new_position[1] - self.agent_position[1]) * (0.5 - self.agent_position[1]), 0.5 + 0.001]
        
        
        self.agent_position = np.clip(np.array(new_position), 0,1)

        self.draw(self.world, self.agent_position, (255, 0, 0))
        self.draw(self.world_copy, self.agent_position, (0, 0, 255))
        

        # Compute the reward
        reward = - np.linalg.norm(self.agent_position - self.goal_position)

        done = self.num_steps >= self.max_steps

        infos = self._get_info()

        return self.agent_position, reward, done, None, infos

    def render(self):
        if self.render_mode == 'rgb_array':
            return self.world.copy()
        else:
            raise NotImplementedError()
        