# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import random
from gym import error, logger, spaces
from ..utils.wrappers import NormalizedBoxEnv
from gym.envs.mujoco import mujoco_env
import torch

# Define the RL environment
class RectangleEnv:
    def __init__(self, data, x_coordinates):
        #self.x_min = -10
        #self.x_max = 10
        #self.y_min = 0
        #self.y_max = 20
        self.x_min = -1
        self.x_max = 1
        self.y_min = -2
        self.y_max = 2
        self.tolerance = 1e-3  # Set the tolerance threshold
        self.data = data
        self.x_coordinates = x_coordinates

        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([10]), dtype=np.float32)

        self.max_step_num = 100
        self.step_count = 0

    def seed(self, seed=0):
        mujoco_env.MujocoEnv.seed(self, seed)

    def get_reward(self, predicted_y, true_y):
        # Calculate the reward based on the prediction error
        return -torch.abs(predicted_y - true_y)

    def reset(self):
        index = np.random.choice(self.data.shape[0], size=1)
        state_action = self.data[index]
        state, action = state_action[:,0], state_action[:,1]
        #state = torch.tensor([np.random.choice(self.x_coordinates)])
        return state

    def step(self, state):
        #next_state = torch.tensor([np.random.choice(self.x_coordinates)])
        index = np.random.choice(self.data.shape[0], size=1)
        state_action = self.data[index]
        state, action = state_action[:,0], state_action[:,1]
        self.step_count += 1
        if self.step_count < self.max_step_num:
            done = False
        else:
            done = True
            self.step_count = 0

        return state, 0, done, {}

    def render(self):
        pass

def generate_solid_rectangle_points(x_min, x_max, y_min, y_max, num_points):
    
    points = []
    for _ in range(num_points):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        points.append((x, y))
    return points

def get_rectangle_env(**kwargs):

    num1 = 800
    num2 = 1300
    num3 = 1300
    points1 = generate_solid_rectangle_points(0.3, 0.5, 0.5, 0.6, num1)
    points2 = generate_solid_rectangle_points(0.5, 0.7, 0.3, 0.5, num2)
    points3 = generate_solid_rectangle_points(0.1, 0.3, 0.1, 0.3, num3)

    # Combine all points
    all_points = points1 + points2 + points3

    # Extract x and y coordinates for plotting
    x_coords, y_coords = zip(*all_points)
    x_coords, y_coords = np.array(x_coords), np.array(y_coords)
    #import ipdb; ipdb.set_trace()
    data = np.concatenate((x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)), axis=1)

    #return RectangleEnv(data, x_coordinates)
    return NormalizedBoxEnv(RectangleEnv(data, x_coords))
