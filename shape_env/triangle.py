# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import matplotlib.pyplot as plt

# Set the number of points per cluster
num_points = 500

# Define the coordinates of the triangle vertices
vertex1 = np.array([0, 0])
vertex2 = np.array([1, 0])
vertex3 = np.array([0.5, 0.866])  # Adjust the y-coordinate to control the triangle's height

# Generate random points around each vertex
cluster1 = np.random.randn(num_points, 2) * 0.05 + vertex1
cluster2 = np.random.randn(num_points, 2) * 0.05 + vertex2
cluster3 = np.random.randn(num_points, 2) * 0.05 + vertex3

# Combine the points from all clusters
points = np.concatenate((cluster1, cluster2, cluster3), axis=0)

# Plot the points
plt.scatter(points[:, 0], points[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points forming a Triangle')
plt.savefig('triangle.png')
#plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt

# Set the number of points per cluster
num_points = 100

# Define the center coordinates of the circles
center1 = np.array([0, 0])
center2 = np.array([2, 0])
center3 = np.array([1, 2])

# Define the radii of the circles
radius1 = 1.5
radius2 = 0.8
radius3 = 1.2

# Generate random points around each circle
theta = np.linspace(0, 2*np.pi, num_points)[:, None]

cluster1 = center1 + radius1 * np.hstack((np.cos(theta), np.sin(theta)))
cluster2 = center2 + radius2 * np.hstack((np.cos(theta), np.sin(theta)))
cluster3 = center3 + radius3 * np.hstack((np.cos(theta), np.sin(theta)))

# Combine the points from all clusters
points = np.concatenate((cluster1, cluster2, cluster3), axis=0)

# Plot the points
plt.scatter(points[:, 0], points[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points forming Circles')
plt.axis('equal')  # Equal aspect ratio for better visualization
plt.savefig('triangle.png')
'''
