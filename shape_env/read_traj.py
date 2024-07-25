# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import matplotlib.pyplot as plt

data = torch.load('sine_2000.pt')
states = data['obs']
actions = data['actions']

plt.figure(figsize=(8, 6))
plt.scatter(states, actions, s=10)  # s参数控制散点的大小
plt.title('(State, Action) Pairs')
plt.xlabel('State')
plt.ylabel('Action')
plt.grid(True)
plt.axis('equal')
plt.savefig('sine_2000.png')

# import matplotlib.pyplot as plt
# import numpy as np

# # 创建一个示例数据，你需要将其替换为你的实际数据
# data = np.random.rand(10, 10)  # 10x10的示例数据
# min_x = 0
# max_x = 10
# min_y = 0
# max_y = 10

# plt.imshow(data, origin='lower', extent=[min_x, max_x, min_y, max_y], cmap='viridis', aspect='auto')
# plt.colorbar(label='Frequency')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Density of Division')

# # 保存图像
# plt.savefig('division_plot.png')
# plt.show()
