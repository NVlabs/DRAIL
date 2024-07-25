# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import random
import matplotlib.pyplot as plt
import torch
import numpy as np

def generate_solid_rectangle_points(x_min, x_max, y_min, y_max, num_points):
    """
    Generate a set of points forming a solid rectangle.

    Arguments:
    - x_min: The minimum x coordinate of the rectangle.
    - x_max: The maximum x coordinate of the rectangle.
    - y_min: The minimum y coordinate of the rectangle.
    - y_max: The maximum y coordinate of the rectangle.
    - num_points: The number of points to generate within the rectangle.

    Returns:
    - points: A set of points forming a solid rectangle.
    """
    points = []
    for _ in range(num_points):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        points.append((x, y))
    return points

# Generate points for the three rectangles
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
x = torch.tensor(x_coords).unsqueeze(1)
y = torch.tensor(y_coords).unsqueeze(1)
next_x = torch.cat((x[1:,:], torch.tensor([[2]]).view(-1, 1)), dim=0)

zeros_array = np.zeros(num1 + num2 + num3 - 1)
ones_array = np.ones(1)
dones = torch.from_numpy(np.concatenate((zeros_array, ones_array), axis=0))

torch.save({
        'obs': x,
        'next_obs': next_x,
        'actions': y,
        'done': dones,
        }, 'rectangle_100.pt')

# Plot the combined rectangles
plt.plot(x_coords, y_coords, 'bo')
plt.title('Combined Rectangles')
plt.xlabel('X')
plt.ylabel('Y')
#plt.grid(True)
#plt.show()
plt.savefig('rectangle.png')