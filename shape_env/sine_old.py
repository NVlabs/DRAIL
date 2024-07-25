# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import matplotlib.pyplot as plt

# Generate data from sine function
np.random.seed(42)  # Set random seed for reproducibility

# Define the parameters for the sine function
amplitude = 1.0  # Amplitude of the sine wave
frequency = 0.1  # Frequency of the sine wave
phase = 0.0  # Phase shift of the sine wave
noise_std = 0.05  # Standard deviation of the Gaussian noise

# Generate x values
# x = np.linspace(0, 10, num=100000)
# y = amplitude * np.sin(2 * np.pi * frequency * x + phase) + np.random.normal(0, noise_std, size=len(x))
x = np.linspace(0, 10, num=2000)
y = amplitude * np.sin(2 * np.pi * frequency * x + phase) + np.random.normal(0, noise_std, size=len(x))

plt.figure(figsize=(10, 3))
plt.axis('equal')
plt.scatter(x, y, s=1)
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('sine.png')
