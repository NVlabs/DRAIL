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

epoch = 20 #20

x = torch.Tensor()
next_x = torch.Tensor()
y = torch.Tensor()
dones = torch.Tensor()

# Generate data from sine function
for i in range(epoch):
    np.random.seed(i)
    
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
    epoch_x = points[:, 0]
    epoch_y = points[:, 1]
    #import ipdb; ipdb.set_trace()
    epoch_next_x = np.concatenate((epoch_x[1:], (np.random.randn(1, 2) * 0.05 + vertex1)[0][0:1]), axis=0)

    zeros_array = np.zeros(num_points*3 - 1)  # Create an array of 99 zeros
    ones_array = np.ones(1) 
    epoch_dones = np.concatenate((zeros_array, ones_array), axis=0)

    x = torch.cat((x, torch.from_numpy(epoch_x).view(-1, 1)), dim=0)
    next_x = torch.cat((next_x, torch.from_numpy(epoch_next_x).view(-1, 1)), dim=0)
    y = torch.cat((y, torch.from_numpy(epoch_y).view(-1, 1)), dim=0)
    dones = torch.cat((dones, torch.from_numpy(epoch_dones).view(-1, 1)), dim=0)

    # if i == 0:
    #     plt.scatter(points[:, 0], points[:, 1])
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Points forming a Triangle')
    #     plt.savefig('triangle_2.png')

torch.save({
        'obs': x,
        'next_obs': next_x,
        'actions': y,
        'done': dones,
        }, 'triangle_2_100.pt')

#Plot the points
plt.scatter(points[:, 0], points[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.title('Points forming a Triangle')
plt.savefig('triangle_2_.png')