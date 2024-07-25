# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys

import copy
import gym
import numpy as np
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlf.algos.il.base_il import BaseILAlgo
from rlf.args import str2bool
from rlf.storage.base_storage import BaseStorage
from tqdm import tqdm
import wandb
import math

class DiffPolicy(BaseILAlgo):

    def __init__(self, set_arg_defs=True):
        super().__init__()
        self.set_arg_defs = set_arg_defs

    def init(self, policy, args):
        super().init(policy, args)
        self.num_epochs = 0
        self.action_dim = rutils.get_ac_dim(self.policy.action_space)
        if self.args.bc_state_norm:
            self.norm_mean = self.expert_stats["state"][0]
            self.norm_var = torch.pow(self.expert_stats["state"][1], 2)
        else:
            self.norm_mean = None
            self.norm_var = None
        self.num_bc_updates = 0
        self.L1 = nn.L1Loss().cuda()
        # self.lambda_bc = args.lambda_bc
        # self.lambda_dm = args.lambda_dm
        num_steps = 1000
        dim = 2

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.bc_state_norm:
            print("Setting environment state normalization")
            settings.state_fn = self._norm_state
        return settings

    def _norm_state(self, x):
        obs_x = torch.clamp(
            (rutils.get_def_obs(x) - self.norm_mean)
            / (torch.pow(self.norm_var, 0.5) + 1e-8),
            -10.0,
            10.0,
        )
        if isinstance(x, dict):
            x["observation"] = obs_x
            return x
        else:
            return obs_x

    def get_num_updates(self):
        if self.exp_generator is None:
            return len(self.expert_train_loader) * self.args.bc_num_epochs
        else:
            return self.args.exp_gen_num_trans * self.args.bc_num_epochs

    def get_completed_update_steps(self, num_updates):
        return num_updates * self.args.traj_batch_size

    def _reset_data_fetcher(self):
        super()._reset_data_fetcher()
        self.num_epochs += 1

    def full_train(self, update_iter=0):
        action_loss = []
        diff_loss = []
        prev_num = 0

        # First BC
        with tqdm(total=self.args.bc_num_epochs) as pbar:
            while self.num_epochs < self.args.bc_num_epochs:
                super().pre_update(self.num_bc_updates)
                log_vals = self._bc_step(False)
                action_loss.append(log_vals["_pr_action_loss"])
                diff_loss.append(log_vals["_pr_diff_loss"])

                pbar.update(self.num_epochs - prev_num)
                prev_num = self.num_epochs

        rutils.plot_line(
            action_loss,
            f"action_loss_{update_iter}.png",
            self.args.vid_dir,
            not self.args.no_wb,
            self.get_completed_update_steps(self.update_i),
        )
        self.num_epochs = 0

    def pre_update(self, cur_update):
        # Override the learning rate decay
        pass
    
    def _bc_step(self, decay_lr):
        if decay_lr:
            super().pre_update(self.num_bc_updates)
        expert_batch = self._get_next_data()
        if expert_batch is None:
            self._reset_data_fetcher()
            expert_batch = self._get_next_data()

        states, true_actions = self._get_data(expert_batch)
        #noise_shape = true_actions.size()
        
        log_dict = {}
        #pred_actions = self.policy.predict_action(noise_shape, states, self.args.device)
        
        pred_loss = self.policy.get_loss(true_actions, states)

        self._standard_step(pred_loss) #backward
        self.num_bc_updates += 1

        #val_loss = self._compute_val_loss()
        #if val_loss is not None:
        #    log_dict["_pr_val_loss"] = val_loss.item()
        # 
        log_dict["_pr_predict_loss"] = pred_loss.item()
        
        return log_dict

    def _get_data(self, batch):
        states = batch["state"].to(self.args.device)
        if self.args.bc_state_norm:
            states = self._norm_state(states)

        if self.args.bc_noise is not None:
            add_noise = torch.randn(states.shape) * self.args.bc_noise
            states += add_noise.to(self.args.device)
            states = states.detach()

        true_actions = batch["actions"].to(self.args.device)
        true_actions = self._adjust_action(true_actions)
        return states, true_actions
    '''
    def _compute_val_loss(self):
        if self.update_i % self.args.eval_interval != 0:
            return None
        if self.val_train_loader is None:
            return None
        with torch.no_grad():
            action_losses = []
            diff_losses = []
            for batch in self.val_train_loader:
                states, true_actions = self._get_data(batch)
                pred_actions = self.policy.predict_action(true_actions.size(), states, self.args.device)
                action_loss = autils.compute_ac_loss(
                    pred_actions,
                    true_actions.view(-1, self.action_dim),
                    self.policy.action_space,
                )
                action_losses.append(action_loss.item())
                
                pred_loss = self.get_loss(states, pred_actions)
                pred_density = math.exp(-pred_loss)
                diff_losses.append(pred_loss.item())
            return np.mean(diff_losses)
    '''
    def update(self, storage, args, beginning, t):
        top_log_vals = super().update(storage) #actor_opt_lr
        log_vals = self._bc_step(True) #_pr_action_loss
        log_vals.update(top_log_vals) #_pr_action_loss
        return log_vals

    def get_storage_buffer(self, policy, envs, args):
        return BaseStorage()

    def get_add_args(self, parser):
        if not self.set_arg_defs:
            # This is set when BC is used at the same time as another optimizer
            # that also has a learning rate.
            self.set_arg_prefix("bc")

        super().get_add_args(parser)
        #########################################
        # Overrides
        if self.set_arg_defs:
            parser.add_argument("--num-processes", type=int, default=1)
            parser.add_argument("--num-steps", type=int, default=0)
            ADJUSTED_INTERVAL = 200
            parser.add_argument("--log-interval", type=int, default=ADJUSTED_INTERVAL)
            parser.add_argument(
                "--save-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
            parser.add_argument(
                "--eval-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
        parser.add_argument("--no-wb", default=False, action="store_true")

        #########################################
        # New args
        parser.add_argument("--bc-num-epochs", type=int, default=1)
        parser.add_argument("--bc-state-norm", type=str2bool, default=False)
        parser.add_argument("--bc-noise", type=float, default=None)
        parser.add_argument("--lambda-bc", type=float, default=1)
        parser.add_argument("--lambda-dm", type=float, default=1)
