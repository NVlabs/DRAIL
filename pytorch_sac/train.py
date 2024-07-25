import sys
sys.path.insert(0, './')
import sys
sys.path.insert(0, './d4rl')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import os.path as osp
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger, WbLogger
from replay_buffer import ReplayBuffer
from sqil_replay_buffer import SqilReplayBuffer
import utils
import goal_prox.envs
import goal_prox.envs.ant
import goal_prox.envs.gridworld
import goal_prox.envs.hand
import goal_prox.envs.goal_check
from rlf.envs.env_interface import get_env_interface
import goal_prox.envs.fetch
import goal_prox.envs.d4rl

import hydra
import gym
import wandb
import attr
import d4rl
import rlf.rl.utils as rutils
from rlf.rl.envs import TransposeImage

@attr.s(auto_attribs=True, slots=True)
class ArgsData:
    ant_noise: float
    ant_cover: int
    ant_is_expert: bool
    fetch_obj_range: float
    fetch_goal_range: float
    fetch_ctrl_pen: float
    fetch_easy_obs: bool
    fetch_cover: float
    noise_ratio: float
    goal_noise_ratio: float
    img_dim: int
    gf_dense: bool
    gw_cover: float
    env_name: str

    gw_compl=False
    gw_spec_goal=False
    gw_goal_pos=None
    gw_goal_quad=None
    gw_agent_quad=None
    gw_img=True
    gw_rand_pos=True
    d4rl_cover = 1.0
    pen_easy_obs=False
    hand_dense=True
    hand_inc_goal=True
    hand_end_on_succ=True
    hand_easy=True



class ObsOnlyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space.spaces['observation']
        self._max_episode_steps = rutils.get_env_attr(env, '_max_episode_steps')

    def step(self, a):
        obs, reward, done, info = super().step(a)
        obs = obs['observation']
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        obs = obs['observation']
        return obs


def make_env(cfg):
    env_name = cfg.env
    args = ArgsData(cfg.env_noise, 100, False,
            None, None, 0.0, True, 1.0, cfg.env_noise, cfg.env_noise, None,
            True, cfg.gw_cover, cfg.env)
    env_interface = get_env_interface(env_name)(args)
    env_interface.setup(args, None)
    env = env_interface.create_from_id(env_name)
    env = env_interface.env_trans_fn(env, False)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = ObsOnlyWrapper(env)
    keys = rutils.get_ob_keys(env.observation_space)
    transpose_keys = [k for k in keys
            if len(rutils.get_ob_shape(env.observation_space, k)) == 3]
    if len(transpose_keys) > 0:
        env = TransposeImage(env, op=[2, 0, 1], transpose_keys=transpose_keys)
    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        if cfg.log_wb:
            self.logger = WbLogger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)
        else:
            self.logger = Logger(self.work_dir,
                                 save_tb=cfg.log_save_tb,
                                 log_frequency=cfg.log_frequency,
                                 agent=cfg.agent.name)

        self.video_dir = './data/vids/'
        if cfg.log_wb:
            wandb.run.save()
            self.video_dir = osp.join(self.video_dir, wandb.run.name)
        else:
            self.video_dir = osp.join(self.video_dir, 'debug')
        if not osp.exists(self.video_dir):
            os.makedirs(self.video_dir)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        ac_dim = rutils.get_ac_dim(self.env.action_space)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_dim = ac_dim
        if not rutils.is_discrete(self.env.action_space):
            cfg.agent.params.action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]
        else:
            cfg.agent.params.action_range = None
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.agent.set_config(cfg)

        if cfg.use_sqil:
            if rutils.is_discrete(self.env.action_space):
                ac_shape = (1,)
            else:
                ac_shape = self.env.action_space.shape
            self.replay_buffer = SqilReplayBuffer(self.env.observation_space.shape,
                                              ac_shape,
                                              int(cfg.replay_buffer_capacity),
                                              self.device, cfg.expertfrac,
                                              cfg.agent.params.batch_size,
                                              cfg.expertpath)
        else:
            self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                              self.env.action_space.shape,
                                              int(cfg.replay_buffer_capacity),
                                              self.device)

        flat_dict = {}
        def get_ks(d):
            for k,v in dict(d).items():
                if 'Dict' in str(type(v)):
                    get_ks(v)
                else:
                    flat_dict[k] = v
        get_ks(cfg)

        if cfg.log_wb:
            wandb.config.update(flat_dict)

        self.video_recorder = VideoRecorder(
            self.video_dir if cfg.save_video else None,
            not rutils.is_discrete(self.env.action_space))
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        average_found_goal = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            if 'ep_found_goal' in info:
                average_found_goal += info['ep_found_goal']
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        average_found_goal /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/found_goal', average_found_goal,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        last_eval_step = 0

        info = {}
        start_time = time.time()
        if hasattr(self.env.env, '_max_episode_steps'):
            max_steps = self.env.env._max_episode_steps
        elif not hasattr(self.env, '_max_episode_steps'):
            max_steps = rutils.get_env_attr(self.env, 'max_steps')
        else:
            max_steps = self.env._max_episode_steps
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and (self.step-last_eval_step) > self.cfg.eval_frequency:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    last_eval_step = self.step

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                if 'ep_found_goal' in info:
                    self.logger.log('train/found_goal', info['ep_found_goal'],
                                    self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == max_steps else done
            episode_reward += reward

            if obs.dtype == np.int64:
                obs = obs.astype(np.float32)

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
