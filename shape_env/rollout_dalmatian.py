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

final_x = torch.Tensor()
final_next_x = torch.Tensor()
final_y = torch.Tensor()
final_dones = torch.Tensor()

for j in range(1):
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
    epoch_x = points[:, 0]
    epoch_y = points[:, 1]
    next_ = np.random.uniform(low=-10, high=10, size=1)
    #import ipdb; ipdb.set_trace()
    epoch_next_x = np.concatenate((points[1:, 0], np.array(next_)))

    zeros_array = np.zeros(epoch_x.shape[0])  # Create an array of 99 zeros
    ones_array = np.ones(1) 
    epoch_dones = np.concatenate((zeros_array, ones_array), axis=0)

    final_x = torch.cat((final_x, torch.from_numpy(epoch_x).view(-1, 1)), dim=0)
    final_next_x = torch.cat((final_next_x, torch.from_numpy(epoch_next_x).view(-1, 1)), dim=0)
    final_y = torch.cat((final_y, torch.from_numpy(epoch_y).view(-1, 1)), dim=0)
    final_dones = torch.cat((final_dones, torch.from_numpy(epoch_dones).view(-1, 1)), dim=0)

torch.save({
        'obs': final_x,
        'next_obs': final_next_x,
        'actions': final_y,
        'done': final_dones,
        }, 'dalmatian_1.pt')

'''
import numpy as np
import matplotlib.pyplot as plt

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
import ipdb; ipdb.set_trace()
points = np.array(points)
next_ = np.random.uniform(low=-10, high=10, size=1)
x = points[:, 0]
next_x = np.concatenate((points[1:, 0], np.array(next_)))
y = points[:, 0]

torch.save({
        'obs': x,
        'next_obs': next_x,
        'actions': y,
        'done': dones,
        }, 'sine_100.pt')
'''