# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import os

dir = os.path.dirname(os.path.realpath(__file__))

traj = torch.load(os.path.join(dir, "..", "expert_datasets/ppo_walker_25.pt"))
print(traj["obs"].shape)
print(traj["actions"].shape)
print(traj["done"].sum())

target_list = [1, 2, 3, 5]
res_trajs = {}
for target in target_list:
    res_trajs[target] = {}

for (k, v) in traj.items():
    trajs = v.split(1000, dim=0)
    for target in target_list:
        res_trajs[target][k] = torch.cat(trajs[:target], dim=0)
    
for target, trajs in res_trajs.items():
    print(target)
    torch.save(trajs, os.path.join(dir, "..", f'expert_datasets/ppo_walker_{target}.pt'))
