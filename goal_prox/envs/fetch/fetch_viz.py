# From https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py

import gym
import numpy as np
import argparse
import sys
sys.path.insert(0, './')
from rlf.rl.utils import save_mp4
from rlf.il.traj_mgr import TrajSaver
import torch
import os.path as osp
from tqdm import tqdm
from goal_prox.envs.goal_traj_saver import GoalTrajSaver
import torch.optim as optim
import goal_prox.envs.fetch
import uuid
from rlf.rl.model import def_mlp_weight_init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import rlf.rl.utils as rutils
import matplotlib.pyplot as plt
from goal_prox.envs.goal_check import EasyObsFetchWrapper




"""Data generation for the case of a single block pick and place in Fetch Env"""

def make_env(env_name, obj_range):
    env = gym.make(env_name)
    env.env.obj_range = obj_range
    return env

def main(save_dir, count, render, obj_range, target_range, args):
    env_name = 'FetchViz-v0'

    rnd_folder = str(uuid.uuid4()).split('-')[0]

    env = gym.make(env_name)
    env.env.obj_range = obj_range
    if target_range != 0.0:
        env.env.target_range = target_range

    if args.easy_obs:
        env = EasyObsFetchWrapper(env)
    env.reset()
    print("Reset!")

    all_frames = []
    prev_obs = None

    for ep_i in tqdm(range(count)):
        obs = env.reset()
        episode_frames, traj_obs, traj_action, traj_info, traj_done = goToGoal(env, obs)

        traj_done = torch.tensor([1.0 if done else 0.0 for done in traj_done])
        traj_obs = torch.tensor([obs['observation'] for obs in traj_obs])
        traj_action = torch.tensor(traj_action)

        traj_len = len(traj_done)
        if render:
            all_frames.extend(episode_frames)
    if render:
        viz_folder = './data/vids/viz'
        save_mp4(all_frames, viz_folder, f"ep_{ep_i}", fps=30,
                no_frame_drop=True)


def goToGoal(env, lastObs):
    goal = lastObs['desired_goal']
    objectPos = lastObs['observation'][3:6]
    object_rel_pos = lastObs['observation'][6:9]
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    episode_frames = []
    episode_dones = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)

    def get_info(next_obs, env):
        try:
            is_success = env.env._is_success(
                        next_obs['achieved_goal'],
                        next_obs['desired_goal'])
        except:
            is_success = env.env.env._is_success(
                        next_obs['achieved_goal'],
                        next_obs['desired_goal'])
        return {
                'ep_found_goal': float(is_success),
                'final_obs': next_obs['observation'],
                }

    #object_rel_pos = [-2.0, 2.0, 1.0]
    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        episode_frames.append(env.render('rgb_array'))
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*6

        action[len(action)-1] = 0.05 #open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(get_info(obsDataNew, env))
        episodeObs.append(obsDataNew)
        episode_dones.append(done)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]
    #return episode_frames, episodeObs, episodeAcs, episodeInfo, episode_dones

    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        episode_frames.append(env.render('rgb_array'))
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(get_info(obsDataNew, env))
        episodeObs.append(obsDataNew)
        episode_dones.append(done)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]


    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
        episode_frames.append(env.render('rgb_array'))
        action = [0, 0, 0, 0]
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(get_info(obsDataNew, env))
        episodeObs.append(obsDataNew)
        episode_dones.append(done)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    while True: #limit the number of timesteps in the episode to a fixed duration
        episode_frames.append(env.render('rgb_array'))
        action = [0, 0, 0, 0]
        action[len(action)-1] = -0.005 # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(get_info(obsDataNew, env))
        episodeObs.append(obsDataNew)
        episode_dones.append(done)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

        if timeStep >= env._max_episode_steps: break

    return episode_frames, episodeObs, episodeAcs, episodeInfo, episode_dones




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str,
            default="./data/traj")
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--obj-range', type=float, default=0.15)
    parser.add_argument('--target-range', type=float, default=0.15)
    parser.add_argument('--env-name', type=str, default='')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--easy-obs', action='store_true')
    args = parser.parse_args()
    main(args.save_dir, args.count, args.render, args.obj_range,
            args.target_range, args)
