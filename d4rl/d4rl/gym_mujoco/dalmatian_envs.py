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
class DalmatianEnv:
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

def get_dalmatian_env(**kwargs):

    # Set the number of circles and the radius range
    num_circles = 15
    min_radius = 0.16
    max_radius = 0.6

    # Generate random center coordinates for the circles
    center_x = np.random.uniform(low=-4, high=4, size=num_circles)
    center_y = np.random.uniform(low=-4, high=4, size=num_circles)

    # Generate random radii for the circles
    radii = np.random.uniform(low=min_radius, high=max_radius, size=num_circles)

    # Generate points within each circle
    
    points = []
    for i in range(num_circles):
        if radii[i] > 0.48:
            num_points = 250 # Number of points per circle
        elif radii[i] > 0.36:
            num_points = 144
        elif radii[i] > 0.24:
            num_points = 120
        elif radii[i] > 0.12:
            num_points = 96
        else:
            num_points = 80

        theta = np.linspace(0, 2 * np.pi, num_points)
        r = np.random.uniform(low=0, high=radii[i], size=num_points)
        x = center_x[i] + r * np.cos(theta)
        y = center_y[i] + r * np.sin(theta)
        points.extend(np.column_stack((x, y)))
        
    points = np.array(points)
    x = points[:, 0]
   
    #return DalmatianEnv(data, x_coordinates)
    return NormalizedBoxEnv(DalmatianEnv(points, x))
