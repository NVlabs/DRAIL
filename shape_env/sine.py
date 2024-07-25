# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import matplotlib.pyplot as plt
import numpy as np

# sine
f = 1
x = np.linspace(0, 1, 1000, endpoint=True)
y = (np.sin(2 * np.pi * f * x))
plt.plot(x, y)
# plt.show()

# collect data
scale = 20
s = np.repeat(x, scale)
a = np.repeat(y, scale)
noise = np.random.normal(0, 0.2, a.shape)
a_noise = a + noise
plt.scatter(s, a_noise, s=3, alpha=0.02)
plt.savefig('sine.png')