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
class TriangleEnv2:
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

def get_triangle_v2_env(**kwargs):

    # Set the number of points per cluster
    num_points = 500

    # Define the coordinates of the triangle vertices
    vertex1 = np.array([-3.6, -3.5])
    vertex2 = np.array([3.8, 0.2])
    vertex3 = np.array([0.2, 3.9])  # Adjust the y-coordinate to control the triangle's height

    # Generate random points around each vertex
    cluster1 = np.random.randn(num_points, 2) * 0.3 + vertex1
    cluster2 = np.random.randn(num_points, 2) * 0.3 + vertex2
    cluster3 = np.random.randn(num_points, 2) * 0.3 + vertex3

    # Combine the points from all clusters
    points = np.concatenate((cluster1, cluster2, cluster3), axis=0)

    points = np.array(points)
    x = points[:, 0]

    #return TriangleEnv(data, x_coordinates)
    return NormalizedBoxEnv(TriangleEnv2(points, x))
