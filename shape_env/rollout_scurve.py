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
from sklearn.datasets import make_s_curve

num_sample = 5000

data, _ = make_s_curve(n_samples=num_sample, noise=0.1)
x = data[:, 0]
y = data[:, 2]
next_x = np.concatenate((x[1:], np.array([make_s_curve(n_samples=100, noise=0.1)[0][0,0]])), axis=0)

zeros_array = np.zeros(num_sample-1)  # Create an array of 99 zeros
ones_array = np.ones(1) 
dones = np.concatenate((zeros_array, ones_array), axis=0)
    
# plt.figure()
plt.figure()
plt.scatter(data[:, 0], data[:, 2], label='Ground Truth')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('s_curve.png')

x = torch.from_numpy(x).view(-1, 1)
next_x = torch.from_numpy(next_x).view(-1, 1)
y = torch.from_numpy(y).view(-1, 1)
dones = torch.from_numpy(dones).view(-1, 1)

torch.save({
        'obs': x,
        'next_obs': next_x,
        'actions': y,
        'done': dones,
        }, 'scurve_5000.pt')


'''
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve

epoch = 20

x = torch.Tensor()
next_x = torch.Tensor()
y = torch.Tensor()
dones = torch.Tensor()

# Generate data from sine function
for i in range(epoch):
    np.random.seed(i)

    data, _ = make_s_curve(n_samples=100, noise=0.1)
    epoch_x = data[:, 0]
    epoch_y = data[:, 2]
    epoch_next_x = np.concatenate((epoch_x[1:], np.array([make_s_curve(n_samples=100, noise=0.1)[0][0,0]])), axis=0)

    zeros_array = np.zeros(99)  # Create an array of 99 zeros
    ones_array = np.ones(1) 
    epoch_dones = np.concatenate((zeros_array, ones_array), axis=0)
    
    x = torch.cat((x, torch.from_numpy(epoch_x).view(-1, 1)), dim=0)
    next_x = torch.cat((next_x, torch.from_numpy(epoch_next_x).view(-1, 1)), dim=0)
    y = torch.cat((y, torch.from_numpy(epoch_y).view(-1, 1)), dim=0)
    dones = torch.cat((dones, torch.from_numpy(epoch_dones).view(-1, 1)), dim=0)

# Plot the points
# plt.figure()
# plt.scatter(x.squeeze().numpy(), y.squeeze().numpy(), label='Ground Truth')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig('s_curve.png')

plt.figure()
plt.scatter(data[:, 0], data[:, 2], label='Ground Truth')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('s_curve.png')

torch.save({
        'obs': x,
        'next_obs': next_x,
        'actions': y,
        'done': dones,
        }, 'rectangle_100.pt')

'''