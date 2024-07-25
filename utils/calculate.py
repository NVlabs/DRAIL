# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

T = 100
alpha_bar_threshold = 1e-5

# Calculate the desired value for beta_T based on alpha_bar_threshold
desired_alpha_bar_T = alpha_bar_threshold ** (1.0 / T)
beta_T = 1 - desired_alpha_bar_T

# Calculate the linearly scheduled beta_t values
beta_t_values = np.linspace(0, beta_T, T)

# Calculate the corresponding alpha_t values
alpha_t_values = 1 - beta_t_values

# Calculate alpha_bar_t values
alpha_bar_t_values = np.cumprod(alpha_t_values)

print(alpha_bar_t_values)
