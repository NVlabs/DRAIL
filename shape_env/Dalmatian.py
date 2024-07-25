# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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

points = np.array(points)

# Plot the points
plt.scatter(points[:, 0], points[:, 1], color='black', edgecolors='none')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Disjoint Solid Circle Points')
plt.savefig('Dalmatian Spots.png')


### bigger
'''
import numpy as np
import matplotlib.pyplot as plt

# Set the number of circles and the radius range
num_circles = 10
min_radius = 0.2
max_radius = 1.0

# Generate random center coordinates for the circles
center_x = np.random.uniform(low=-10, high=10, size=num_circles)
center_y = np.random.uniform(low=-10, high=10, size=num_circles)

# Generate random radii for the circles
radii = np.random.uniform(low=min_radius, high=max_radius, size=num_circles)

# Generate points within each circle
num_points = 100  # Number of points per circle
points = []
labels = []
for i in range(num_circles):
    theta = np.random.uniform(low=0, high=2 * np.pi, size=num_points)
    r = np.random.uniform(low=0, high=radii[i], size=num_points)
    x = center_x[i] + r * np.cos(theta)
    y = center_y[i] + r * np.sin(theta)
    points.extend(np.column_stack((x, y)))
    labels.extend([i] * num_points)

points = np.array(points)
labels = np.array(labels)

# Plot the points
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab10')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Disjoint Solid Circle Points')
plt.savefig('Dalmatian Spots.png')
'''

'''
import numpy as np
import matplotlib.pyplot as plt

# Set the number of circles and the radius range
num_circles = 10
min_radius = 0.2
max_radius = 1.0

# Generate random center coordinates for the circles
center_x = np.random.uniform(low=-10, high=10, size=num_circles)
center_y = np.random.uniform(low=-10, high=10, size=num_circles)

# Generate random radii for the circles
radii = np.random.uniform(low=min_radius, high=max_radius, size=num_circles)

# Generate points within each circle
num_points = 100  # Number of points per circle
points = []
labels = []
for i in range(num_circles):
    theta = np.random.uniform(low=0, high=2 * np.pi, size=num_points)
    r = np.random.uniform(low=0, high=radii[i], size=num_points)
    x = center_x[i] + r * np.cos(theta)
    y = center_y[i] + r * np.sin(theta)
    points.extend(np.column_stack((x, y)))
    labels.extend([i] * num_points)

points = np.array(points)
labels = np.array(labels)

# Plot the points
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab10')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Disjoint Solid Circle Points')
plt.savefig('Dalmatian Spots.png')
'''


# import numpy as np
# import matplotlib.pyplot as plt

# # Set the number of circles and the radius range
# num_circles = 50
# min_radius = 0.2
# max_radius = 1.0

# # Generate random center coordinates for the circles
# center_x = np.random.uniform(low=-10, high=10, size=num_circles)
# center_y = np.random.uniform(low=-10, high=10, size=num_circles)

# # Generate random radii for the circles
# radii = np.random.uniform(low=min_radius, high=max_radius, size=num_circles)

# # Plot the circles
# fig, ax = plt.subplots(figsize=(8, 8))
# for i in range(num_circles):
#     circle = plt.Circle((center_x[i], center_y[i]), radii[i], edgecolor='black', facecolor='black')
#     ax.add_artist(circle)

# ax.set_xlim(-12, 12)
# ax.set_ylim(-12, 12)
# ax.set_aspect('equal')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Disjoint Solid Circles')
# plt.savefig('Dalmatian Spots.png')
