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
class Sine5Env:
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
        #state = torch.tensor(state).to(torch.float32).reshape(-1,1)
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
        #state = torch.tensor(state).to(torch.float32).reshape(-1,1)
        return state, 0, done, {}

    def render(self):
        pass

def get_sine_5_env(**kwargs):

    # Generate data from sine function
    np.random.seed(42)  # Set random seed for reproducibility
    # Define the parameters for the sine function
    amplitude = 1.0  # Amplitude of the sine wave
    frequency = 0.1  # Frequency of the sine wave
    phase = 0.0  # Phase shift of the sine wave
    noise_std = 0.05  # Standard deviation of the Gaussian noise
    # Generate x values
    x_ = np.linspace(0, 10, num=1000)
    y_offsets = np.linspace(0, 2, num=5)
    x = np.array([])
    y = np.array([])
    for y_offset in y_offsets:
        y_ = amplitude * np.sin(2 * np.pi * frequency * x_ + phase) + np.random.normal(0, noise_std, size=len(x_)) + y_offset
        x = np.concatenate((x, x_))  # Concatenate x values
        y = np.concatenate((y, y_))  # Concatenate y values
    data = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

    return NormalizedBoxEnv(Sine5Env(data, x))
