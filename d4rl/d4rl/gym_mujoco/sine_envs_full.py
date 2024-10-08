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
class SineEnv:
    def __init__(self, get_data):
        #self.x_min = -10
        #self.x_max = 10
        #self.y_min = 0
        #self.y_max = 20
        self.x_min = 0
        self.x_max = 1
        self.y_min = -2
        self.y_max = 2
        self.tolerance = 1e-3  # Set the tolerance threshold
        self.get_data = get_data
        self.data = self.get_data()

        self.action_space = spaces.Box(low=np.array([-2.0]), high=np.array([2.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)

        self.max_step_num = 1000
        self.step_count = 0

    def seed(self, seed=0):
        mujoco_env.MujocoEnv.seed(self, seed)

    def get_reward(self, predicted_y, true_y):
        # Calculate the reward based on the prediction error
        return -torch.abs(predicted_y - true_y)

    def reset(self):
        self.data = self.get_data()
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

def get_sine_env_full(**kwargs):

    def get_data():
        # Define the parameters for the sine function
        amplitude = 1.0  # Amplitude of the sine wave
        frequency = 1  # Frequency of the sine wave
        phase = 0.0  # Phase shift of the sine wave
        noise_std = 0.2  # Standard deviation of the Gaussian noise
        # Generate x values
        x = np.linspace(0, 1, num=10000, endpoint=True)
        y = amplitude * np.sin(2 * np.pi * frequency * x + phase)
        y = y + np.random.normal(0, noise_std, y.shape)
        data = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        return data

    # Initialize the environment
    #state_dim = 1
    #action_dim = 1
    #input_size = state_dim + action_dim
    #earning_rate = 0.01
    #return SineEnv(data, x_coordinates)
    return NormalizedBoxEnv(SineEnv(get_data), scale=2)
