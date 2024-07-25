# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from rlf.algos.il.base_irl import BaseIRLAlgo
import torch
import torch.nn as nn
import torch.nn.functional as F
import rlf.rl.utils as rutils
import rlf.algos.utils as autils
from collections import defaultdict
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.args import str2bool
import torch.optim as optim
import numpy as np
from rlf.rl.model import ConcatLayer
from rlf.rl.model import InjectNet
from functools import partial
from rlf.exp_mgr.viz_utils import append_text_to_image


def get_default_discrim():
    """
    - ac_dim: int will be 0 if no action are used.
    Returns: (nn.Module) Should take state AND actions as input if ac_dim
    != 0. If ac_dim = 0 (discriminator does not use actions) then ONLY take
    state as input.
    """
    hidden_dim = 64
    layers = [
            #nn.Linear(in_shape[0] + ac_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
            ]

    return nn.Sequential(*layers), hidden_dim

class WAIL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([WailDiscrim(get_discrim), agent_updater], 1)

class WailDiscrim(BaseIRLAlgo):
    def __init__(self, get_discrim=None):
        super().__init__()
        if get_discrim is None:
            get_discrim = get_default_discrim
        self.get_discrim = get_discrim
        #import ipdb; ipdb.set_trace()

    def _create_discrim(self):
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        ac_dim = rutils.get_ac_dim(self.action_space)
        base_net = self.policy.get_base_net_fn(ob_shape)
        discrim, dhidden_dim = self.get_discrim()
        discrim_head = InjectNet(
            base_net.net,
            discrim,
            base_net.output_shape[0], dhidden_dim, ac_dim,
            self.args.action_input)

        return discrim_head.to(self.args.device)

    def init(self, policy, args):
        super().init(policy, args)
        self.action_space = self.policy.action_space

        self.discrim_net = self._create_discrim()

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.opt = optim.SGD(
            self.discrim_net.parameters(), lr=self.args.disc_lr)

    def _get_sampler(self, storage):
        agent_experience = storage.get_generator(None,
                                                 mini_batch_size=self.expert_train_loader.batch_size)
        return self.expert_train_loader, agent_experience

    def _trans_batches(self, expert_batch, agent_batch):
        return expert_batch, agent_batch

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if not args.wail_state_norm:
            settings.ret_raw_obs = True
        settings.mod_render_frames_fn = self.mod_render_frames
        return settings

    def mod_render_frames(self, frame, env_cur_obs, env_cur_action, env_cur_reward,
            env_next_obs, **kwargs):
        use_cur_obs = rutils.get_def_obs(env_cur_obs)
        use_cur_obs = torch.FloatTensor(use_cur_obs).unsqueeze(0).to(self.args.device)

        if env_cur_action is not None:
            use_action = torch.FloatTensor(env_cur_action).unsqueeze(0).to(self.args.device)
            disc_val = self._compute_disc_val(use_cur_obs, use_action).item()
        else:
            disc_val = 0.0

        frame = append_text_to_image(frame, [
            "Discrim: %.3f" % disc_val,
            "Reward: %.3f" % (env_cur_reward if env_cur_reward is not None else 0.0)
            ])
        return frame

    def _norm_expert_state(self, state, obsfilt):
        if not self.args.wail_state_norm:
            return state
        state = state.cpu().numpy()

        if obsfilt is not None:
            state = obsfilt(state, update=False)
        state = torch.tensor(state).to(self.args.device)
        return state

    def _trans_agent_state(self, state, other_state=None):
        if not self.args.wail_state_norm:
            if other_state is None:
                return state['raw_obs']
            return other_state['raw_obs']
        return rutils.get_def_obs(state)

    def _compute_discrim_loss(self, agent_batch, expert_batch, obsfilt):
        expert_actions = expert_batch['actions'].to(self.args.device)
        expert_actions = self._adjust_action(expert_actions)
        expert_states = self._norm_expert_state(expert_batch['state'],
                obsfilt)

        agent_states = self._trans_agent_state(agent_batch['state'],
                agent_batch['other_state'] if 'other_state' in agent_batch else None)
        agent_actions = agent_batch['action']

        agent_actions = rutils.get_ac_repr(
            self.action_space, agent_actions)
        expert_actions = rutils.get_ac_repr(
            self.action_space, expert_actions)

        expert_d = self._compute_disc_val(expert_states, expert_actions)
        agent_d = self._compute_disc_val(agent_states, agent_actions)

        grad_pen = self.compute_pen(expert_states, expert_actions, agent_states,
                agent_actions)

        return expert_d, agent_d, grad_pen

    def compute_pen(self, expert_states, expert_actions, agent_states, agent_actions):
        grad_pen = self.args.disc_grad_pen * autils.wass_grad_pen(expert_states,
                expert_actions, agent_states, agent_actions,
                self.args.action_input, self._compute_disc_val)
        return grad_pen

    def _compute_disc_val(self, state, action):
        return self.discrim_net(state, action)

    def _compute_expert_loss(self, expert_d, expert_batch):
        return expert_d.mean()
        # return F.binary_cross_entropy_with_logits(expert_d,
        #         torch.ones(expert_d.shape).to(self.args.device))

    def _compute_agent_loss(self, agent_d, agent_batch):
        return - agent_d.mean()
        # return F.binary_cross_entropy_with_logits(agent_d,
        #         torch.zeros(agent_d.shape).to(self.args.device))

    def _update_reward_func(self, storage, gradient_clip=True, t=1):
        self.discrim_net.train()

        log_vals = defaultdict(lambda: 0)
        obsfilt = self.get_env_ob_filt()

        n = 0
        expert_sampler, agent_sampler = self._get_sampler(storage)
        if agent_sampler is None:
            # algo requested not to update this step
            return {}

        for epoch_i in range(self.args.n_wail_epochs):
            for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
                expert_batch, agent_batch = self._trans_batches(
                    expert_batch, agent_batch)
                n += 1
                expert_d, agent_d, grad_pen = self._compute_discrim_loss(agent_batch, expert_batch, obsfilt)
                expert_loss = self._compute_expert_loss(expert_d, expert_batch)
                agent_loss = self._compute_agent_loss(agent_d, agent_batch)

                discrim_loss = expert_loss + agent_loss

                if self.args.disc_grad_pen != 0.0:
                    log_vals['grad_pen'] += grad_pen.item()
                    total_loss = discrim_loss + grad_pen
                else:
                    total_loss = discrim_loss

                torch.nn.utils.clip_grad_norm_(self.discrim_net.parameters(), max_norm=1)
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

                log_vals['discrim_loss'] += discrim_loss.item()
                log_vals['expert_loss'] += expert_loss.item()
                log_vals['agent_loss'] += agent_loss.item()

        for k in log_vals:
            log_vals[k] /= n

        return log_vals

    def _compute_discrim_reward(self, storage, step, add_info):
        state = self._trans_agent_state(storage.get_obs(step))
        action = storage.actions[step]
        action = rutils.get_ac_repr(self.action_space, action)
        d_val = self._compute_disc_val(state, action)
        s = torch.sigmoid(d_val)
        eps = 1e-20
        if self.args.reward_type == 'wail':
            #reward = -s
            reward = - d_val
        else:
            raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        return reward

    def _get_reward(self, step, storage, add_info):
        masks = storage.masks[step]
        with torch.no_grad():
            self.discrim_net.eval()
            reward = self._compute_discrim_reward(storage, step, add_info)

            if self.args.wail_reward_norm:
                if self.returns is None:
                    self.returns = reward.clone()

                self.returns = self.returns * masks * self.args.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), {}
            else:
                return reward, {}

    def eval_disc(self, state, action):
        self.discrim_net.eval()
        with torch.no_grad():
            discrim_output = self.discrim_net(state, action)
        return -discrim_output

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides

        #########################################
        # New args
        parser.add_argument('--action-input', type=str2bool, default=False)
        parser.add_argument('--wail-reward-norm', type=str2bool, default=False)
        parser.add_argument('--wail-state-norm', type=str2bool, default=True)
        parser.add_argument('--disc-lr', type=float, default=0.0001)
        parser.add_argument('--disc-grad-pen', type=float, default=0.0)
        parser.add_argument('--n-wail-epochs', type=int, default=1)
        parser.add_argument('--reward-type', type=str, default='wail', help="""
                One of [airl, raw, wail]. Changes the reward computation. Does
                not change training.
                """)

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        self.opt.load_state_dict(checkpointer.get_key('wail_disc_opt'))
        self.discrim_net.load_state_dict(checkpointer.get_key('wail_disc'))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key('wail_disc_opt', self.opt.state_dict())
        checkpointer.save_key('wail_disc', self.discrim_net.state_dict())
