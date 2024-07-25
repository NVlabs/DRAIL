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
from sklearn.datasets import make_s_curve

# Define the RL environment
class SCurveEnv:
    def __init__(self, data, x_coordinates):

        self.data = data
        self.x_coordinates = x_coordinates
        
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([10]), dtype=np.float32)

    def seed(self, seed=0):
        mujoco_env.MujocoEnv.seed(self, seed)

    def get_reward(self, predicted_y, true_y):
        # Calculate the reward based on the prediction error
        return -torch.abs(predicted_y - true_y)

    def reset(self):
        state = torch.tensor([np.random.choice(self.x_coordinates)])
        # Reset the environment
        return state

    def step(self, state):
        # Get the true y-coordinate for the given x-coordinate
        next_state = torch.tensor([np.random.choice(self.x_coordinates)])
        done = False

        return next_state, 0, done, {}

    def render(self):
        pass

def get_scurve_env(**kwargs):

    # Generate data from sine function
    np.random.seed(42)  # Set random seed for reproducibility
    data, _ = make_s_curve(n_samples=10000, noise=0.1)
    x = data[:, 0]
    y = data[:, 2]
    data = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    #import ipdb; ipdb.set_trace()

    # Initialize the environment
    state_dim = 1
    action_dim = 1
    input_size = state_dim + action_dim
    learning_rate = 0.01

    #return SineEnv(data, x_coordinates)
    return NormalizedBoxEnv(SCurveEnv(data, x))
