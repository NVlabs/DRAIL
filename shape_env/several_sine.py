# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import matplotlib.pyplot as plt
import torch

# Generate data from parallel sine functions
np.random.seed(42)  # Set random seed for reproducibility

# Define the parameters for the sine functions
amplitude = 1.0  # Amplitude of the sine waves
frequency = 0.1  # Frequency of the sine waves
phase = 0.0  # Phase shift of the sine waves
sine_num = 5
x_num = 10000
# y_offsets = np.linspace(0, 2, num=sine_num)
y_offsets = np.linspace(-5, 5, num=sine_num)  # Y-axis offsets for the sine waves
noise_std = 0.05  # Standard deviation of the Gaussian noise

# Generate x values
x = np.linspace(0, 10, num=x_num)  # Adjust the range and number of samples as needed

# Generate y values for each sine function with noise and y-axis offset
x_values = torch.Tensor()
y_values = torch.Tensor()
y_list = []
for y_offset in y_offsets:
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase) + np.random.normal(0, noise_std, size=len(x)) + y_offset
    y_list.append(y)
    x_values = torch.cat((x_values, torch.from_numpy(x).view(-1, 1)), dim=0)
    y_values = torch.cat((y_values, torch.from_numpy(y).view(-1, 1)), dim=0)

# Plot the generated data for all sine functions
plt.figure()
for i, y in enumerate(y_list):
    plt.scatter(x, y, color='#ADD8E6', s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('several_sines_v2.png')

zeros = torch.zeros(sine_num * x_num - 1)
ones = torch.ones(1)
dones = torch.cat((zeros, ones), dim=0)
next_x = torch.cat((x_values[1:, :], torch.from_numpy(np.linspace(0, 10, num=1)).view(-1, 1)), dim=0)
#import ipdb; ipdb.set_trace()

torch.save({
        'obs': x_values,
        'next_obs': next_x,
        'actions': y_values,
        'done': dones,
        }, '5_sine_100_v2.pt')
