# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(0)

x = torch.Tensor()
next_x = torch.Tensor()
y = torch.Tensor()
dones = torch.Tensor()

amplitude = 1.0  # Amplitude of the sine wave
frequency = 0.1  # Frequency of the sine wave
phase = 0.0  # Phase shift of the sine wave
noise_std = 0.05  # Standard deviation of the Gaussian noise

dir = os.path.dirname(os.path.realpath(__file__))

def random_sample_from_intervals(intervals):
        interval = intervals[torch.randint(len(intervals), (1,))]
        return torch.rand(1) * (interval[1] - interval[0]) + interval[0]

def sample_expert_ground_truth(num, min=0, max=10, split=100):
        intervals = np.arange(min, max, (max-min)/split).reshape(-1, 2).tolist()
        x = torch.stack(
                [random_sample_from_intervals(intervals) for _ in range(num)]
        ).squeeze()
        return x

def plot_graph(x, y):
    # Plot the graph
    plt.scatter(x, y, s=3, alpha=0.1)

    x = np.linspace(0, 1, 1000, endpoint=True)
    y = (np.sin(2 * np.pi * frequency * x))
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.ylim((-2, 2))
    plt.title('Sine Expert Distribution')

    # Show the plot
    plt.savefig(os.path.join(dir, "../expert_datasets", "sine.png"))


# epoch_x = np.linspace(0, 10, num=2000)
epoch_x = sample_expert_ground_truth(2000, 0, 10, 20).numpy()
epoch_y = amplitude * np.sin(2 * np.pi * frequency * epoch_x + phase) + np.random.normal(0, noise_std, size=len(epoch_x))
epoch_next_x = np.concatenate((epoch_x[1:], np.linspace(0, 10, num=1)), axis=0)

zeros_array = np.zeros(99)  # Create an array of 99 zeros
ones_array = np.ones(1) 
epoch_dones = np.concatenate((zeros_array, ones_array), axis=0).repeat(20)

x = torch.cat((x, torch.from_numpy(epoch_x).view(-1, 1)), dim=0)
next_x = torch.cat((next_x, torch.from_numpy(epoch_next_x).view(-1, 1)), dim=0)
y = torch.cat((y, torch.from_numpy(epoch_y).view(-1, 1)), dim=0)
dones = torch.cat((dones, torch.from_numpy(epoch_dones).view(-1, 1)), dim=0)

torch.save({
        'obs': x,
        'next_obs': next_x,
        'actions': y,
        'done': dones,
        }, os.path.join(dir, "../expert_datasets", 'sine.pt'))

plot_graph(x, y)