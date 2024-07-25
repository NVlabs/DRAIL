# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Stand alone script to generate the D4RL maze2d dataset in our format.
"""
import sys

sys.path.insert(0, "./")
import gym
import argparse
import os.path as osp
import numpy as np
import d4rl

import goal_prox.envs.ball_in_cup
import goal_prox.envs.cartpole
import goal_prox.envs.d4rl
import goal_prox.envs.fetch
import goal_prox.envs.goal_check
import goal_prox.envs.gridworld
import goal_prox.envs.hand
import goal_prox.gym_minigrid
from rlf.exp_mgr.viz_utils import save_mp4
import torch

# from pyvirtualdisplay import Display

# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()


ENV = "CartPoleCustom-v0"
SAVE_DIR = "expert_video"

def str2bool(v):
    return v.lower() == "true"

def reset_data():
    return {
        "obs": [],
        "next_obs": [],
        "actions": [],
        "done": [],
    }


def append_data(data, s, ns, a, done):
    data["obs"].append(s)
    data["next_obs"].append(ns)
    data["actions"].append(a)
    data["done"].append(done)


def extend_data(data, episode):
    data["obs"].extend(episode["obs"])
    data["next_obs"].extend(episode["next_obs"])
    data["actions"].extend(episode["actions"])
    data["done"].extend(episode["done"])


def npify(data):
    for k in data:
        if k == "dones":
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name", type=str, default=ENV, help="Env name"
    )
    parser.add_argument(
        "--expert-traj", type=str, default="./expert_datasets/cartpole.pt", help="Expert traj"
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--discrete-action", type=str2bool, default=True)
    args = parser.parse_args()

    env = gym.make(args.env_name)

    env.seed(args.seed)
    np.random.seed(args.seed)

    expert_traj = torch.load(args.expert_traj)
    env.reset()
    frames = [env.render("rgb_array")]
    
    for idx in range(min(expert_traj["obs"].size(0), args.max_frames)):
        
        act = expert_traj['actions'][idx]
        
        if args.discrete_action:
            # for discrete action
            act = act.item()
        else:
            # for continuous action
            act = np.asarray(act)
        
        r, _, _, _ = env.step(act)
        
        frames.append(env.render("rgb_array"))
        if expert_traj['done'][idx].item() == True:
            
            env.reset()
            frames.append(env.render("rgb_array"))
        
    save_mp4(frames, SAVE_DIR, args.env_name + "_%d" % idx, fps=60, no_frame_drop=True)
    
    env.close()
    
if __name__ == "__main__":
    main()
