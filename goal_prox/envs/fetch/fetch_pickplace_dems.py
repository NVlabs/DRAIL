# From https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py

import gym
import numpy as np
import argparse
import sys
sys.path.insert(0, './')
from rlf.exp_mgr.viz_utils import save_mp4, save_agent_obs
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
from goal_prox.envs.goal_check import EasyObsFetchWrapper, SingleFrameStack
from rlf.envs.image_obs_env import ImageObsWrapper
from rlf.baselines.common.atari_wrappers import WarpFrame
from rlf.rl.envs import TransposeImage
import goal_prox.envs.viz



"""Data generation for the case of a single block pick and place in Fetch Env"""

def make_env(env_name, obj_range):
    env = gym.make(env_name)
    env.env.obj_range = obj_range
    return env

def get_state(obs):
    if 'state' in obs:
        return obs['state']
    else:
        return obs['observation']

def main(save_dir, count, render, obj_range, target_range, args):
    if target_range == 0.0:
        env_name = 'FetchPickAndPlaceCustom-v0'
    elif args.env_name == 'diff':
        env_name = 'FetchPickAndPlaceDiff-v0'
    elif args.env_name == 'holdout':
        env_name = 'FetchPickAndPlaceDiffHoldout-v0'
    elif args.env_name == 'viz':
        env_name = 'VizFetchPickAndPlaceCustom-v0'
    else:
        env_name = 'FetchPickAndPlaceHarder-v0'

    rnd_folder = str(uuid.uuid4()).split('-')[0]
    traj_saver = GoalTrajSaver(osp.join('./data/traj/',
        rnd_folder, env_name), False)

    env = gym.make(env_name)
    env.env.obj_range = obj_range
    if target_range != 0.0:
        env.env.target_range = target_range

    env.set_noise_ratio(args.noise_ratio, args.goal_noise_ratio)

    if args.env_name == 'holdout':
        env.env.coverage = args.cover

    if args.easy_obs:
        env = EasyObsFetchWrapper(env)
    if args.img_dim is not None:
        env = ImageObsWrapper(env, args.img_dim)
        env = WarpFrame(env, grayscale=True)
        keys = rutils.get_ob_keys(env.observation_space)
        transpose_keys = [k for k in keys
                if len(rutils.get_ob_shape(env.observation_space, k)) == 3]
        env = TransposeImage(env, op=[2, 0, 1], transpose_keys=transpose_keys)
        env = SingleFrameStack(env, 4, 'observation')
    env.reset()

    all_frames = []
    prev_obs = None
    viz_folder = './data/vids/viz'

    for ep_i in tqdm(range(count)):
        obs = env.reset()
        if args.fail:
            override_goal = np.array([1.2, 1.2, 0.6])
        else:
            override_goal = None
        episode_frames, traj_obs, traj_action, traj_info, traj_done = goToGoal(env, obs, override_goal)

        traj_done = torch.tensor([1.0 if done else 0.0 for done in traj_done])
        traj_obs = torch.tensor([obs['observation'] for obs in traj_obs])
        traj_action = torch.tensor(traj_action)

        if render and args.img_dim is not None:
            save_agent_obs(traj_obs, 1, viz_folder, f"ep_{ep_i}")

        traj_len = len(traj_done)
        for i in range(traj_len):
            traj_saver.collect(
                    traj_obs[i].unsqueeze(0),
                    traj_obs[i+1].unsqueeze(0),
                    traj_done[i].unsqueeze(0),
                    traj_action[i].unsqueeze(0),
                    [traj_info[i]])

        if render:
            all_frames.extend(episode_frames)
    if render:
        save_mp4(all_frames, viz_folder, f"ep_{ep_i}", fps=30,
                no_frame_drop=True)
        raise ValueError()

    save_name = traj_saver.save()


def goToGoal(env, last_obs, override_goal=None):
    if override_goal is not None:
        goal = override_goal
    else:
        goal = last_obs['desired_goal']
    objectPos = last_obs['observation'][3:6]
    object_rel_pos = get_state(last_obs)[6:9]
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    episode_frames = []
    episode_dones = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(last_obs)

    def get_info(next_obs, env):
        is_success_fn = rutils.get_env_attr(env, '_is_success')
        is_success = is_success_fn(
                    next_obs['achieved_goal'],
                    next_obs['desired_goal'])
        return {
                'ep_found_goal': float(is_success),
                'final_obs': next_obs['observation'],
                }

    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env.max_episode_steps:
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

        objectPos = get_state(obsDataNew)[3:6]
        object_rel_pos = get_state(obsDataNew)[6:9]

    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env.max_episode_steps :
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

        objectPos = get_state(obsDataNew)[3:6]
        object_rel_pos = get_state(obsDataNew)[6:9]


    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env.max_episode_steps :
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

        objectPos = get_state(obsDataNew)[3:6]
        object_rel_pos = get_state(obsDataNew)[6:9]

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

        objectPos = get_state(obsDataNew)[3:6]
        object_rel_pos = get_state(obsDataNew)[6:9]

        if timeStep >= env.max_episode_steps:
            break

    return episode_frames, episodeObs, episodeAcs, episodeInfo, episode_dones




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str,
            default="./data/traj")
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--obj-range', type=float, default=0.15)
    parser.add_argument('--noise-ratio', type=float, default=1.0)
    parser.add_argument('--goal-noise-ratio', type=float, default=1.0)
    parser.add_argument('--target-range', type=float, default=0.15)
    parser.add_argument('--cover', type=float, default=1.0)
    parser.add_argument('--env-name', type=str, default='')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--fail', action='store_true')
    parser.add_argument('--easy-obs', action='store_true')
    parser.add_argument('--img-dim', type=int, default=None)
    args = parser.parse_args()
    main(args.save_dir, args.count, args.render, args.obj_range,
            args.target_range, args)
