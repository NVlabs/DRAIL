# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from rlf.algos.il.base_irl import BaseIRLAlgo
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler
import rlf.rl.utils as rutils
import rlf.algos.utils as autils
from collections import defaultdict
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.args import str2bool
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from rlf.exp_mgr.viz_utils import append_text_to_image
import math

# Does not necessarily have WB installed
try:
    import wandb
except:
    pass
from drail.ddpm import MLPConditionDiffusion

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, args, base_net, num_units=128):
        super(Discriminator, self).__init__()
        input_dim = state_dim + action_dim
        self.args = args
        self.base_net = False

        self.n_steps = n_steps = 1000
        betas = cosine_beta_schedule(self.n_steps)
        self.betas = betas.to(self.args.device)
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_prod).to(self.args.device)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(self.args.device)

        d_model = MLPConditionDiffusion(n_steps, self.args.label_dim, input_dim, num_units=num_units, depth=self.args.discrim_depth).to(self.args.device)
        try:
            self.base_net = base_net.net.to(self.args.device)
        except:
            self.base_net = False
        self.model = d_model


    def diffusion_loss(self, label, sa_pair, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        batch_size = sa_pair.shape[0]

        t = torch.randint(0, n_steps, size=(batch_size//2,)).to(self.args.device)
        t = torch.cat([t, n_steps-1-t], dim=0) #[batch_size, 1]
        t = t.unsqueeze(-1)
        
        # coefficient of x0
        a = alphas_bar_sqrt[t]
        
        # coefficient of eps
        aml = one_minus_alphas_bar_sqrt[t]
        label_input = torch.full((batch_size, self.args.label_dim), label).to(self.args.device)
        
        # generate random noise eps
        e = torch.randn_like(sa_pair).to(self.args.device)

        # model input
        x = sa_pair*a + e*aml
        
        # get predicted randome noise at time t
        output = self.model(x, label_input, t.squeeze(-1))
        
        return (e - output).square().mean(dim=1, keepdim=True)
        #return torch.unsqueeze(torch.mean(e - output, dim=1), 1)

    def diffusion_loss_fn(self, label, sa_pair):
        diff_loss = self.diffusion_loss(label, sa_pair, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, self.n_steps)
    
        return diff_loss

    def forward(self, state, action, label):
        
        if self.base_net:
            state = self.base_net(state)
        state_action = torch.cat([state, action], dim=1)
        loss = self.diffusion_loss_fn(label, state_action)
        return loss
    
    def p_sample_loop(self, state, action):
        
        if self.base_net:
            state = self.base_net(state)
        cond = torch.cat([state, action], dim=1).to(self.args.device)
        batch_size = cond.shape[0]
        cur_x = torch.randn(batch_size, self.args.label_dim).to(self.args.device)
        x_seq = [cur_x]
        for i in reversed(range(self.n_steps)):
            cur_x = self.p_sample(cur_x,cond,i,self.betas,self.one_minus_alphas_bar_sqrt)
            x_seq.append(cur_x)
        return x_seq

    def p_sample(self, x, c, t, betas,one_minus_alphas_bar_sqrt):
        # sample reconstruction data at time t drom x[T]
        t = torch.tensor([t]).to(self.args.device)

        coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    
        eps_theta = self.model(x,c,t)
    
        mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
        
        z = torch.randn_like(x)
        sigma_t = betas[t].sqrt()
    
        sample = mean + sigma_t * z
    
        return (sample)

def norm_vec(x, mean, std):
    obs_x = torch.clamp((x - mean)
        / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x

#! Modify n_steps & num_units
# def get_default_discrim(state_dim, action_dim, args, base_net, n_steps=1000, num_units=128, clip_range=2.0):
def get_default_discrim(state_dim, action_dim, args, base_net, num_units=128):
    """
    - ac_dim: int will be 0 if no action are used.
    Returns: (nn.Module) Should take state AND actions as input if ac_dim
    != 0. If ac_dim = 0 (discriminator does not use actions) then ONLY take
    state as input.
    """

    #return Discriminator(state_dim, action_dim, args, base_net, n_steps=n_steps, num_units=num_units, clip_range=clip_range)
    return Discriminator(state_dim, action_dim, args, base_net, num_units=num_units)

class DRAIL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([DRAILDiscrim(get_discrim, policy=agent_updater), agent_updater], 1)

class DRAILDiscrim(BaseIRLAlgo):
    def __init__(self, get_discrim=None, policy=None):
        super().__init__()
        if get_discrim is None:
            get_discrim = get_default_discrim
        self.get_discrim = get_discrim
        self.policy = policy
        self.step = 0

    def _create_discrim(self):
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        ac_dim = rutils.get_ac_dim(self.action_space)
        base_net = self.policy.get_base_net_fn(ob_shape)
        #* Change to Diffusion Model
        discrim = self.get_discrim(base_net.output_shape[0], ac_dim, self.args, base_net, num_units=self.args.discrim_num_unit)
        return discrim.to(self.args.device)

    def init(self, policy, args):
        super().init(policy, args)
        self.action_space = self.policy.action_space
        self.label_dim = self.args.label_dim

        self.discrim_net = self._create_discrim()

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.opt = optim.Adam(
            self.discrim_net.parameters(), lr=self.args.disc_lr)
        if self.args.lr_schedule:
            self.scheduler = LambdaLR(self.opt, lr_lambda=self.lr_lambda)

    # Define a function to calculate the learning rate at each epoch
    def lr_lambda(self, epoch):
        lr_start = 0.00005
        lr_end = self.args.disc_lr
        x = epoch / self.args.n_drail_epochs 
        if x < 0.5:
            lr = lr_start + (lr_end - lr_start) * math.sqrt(x)
        else:
            lr = self.args.disc_lr
        return lr

    def adjust_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
    
    def _get_sampler(self, storage):
        agent_experience = storage.get_generator(None,
                                                 mini_batch_size=self.expert_train_loader.batch_size)
        return self.expert_train_loader, agent_experience

    def _trans_batches(self, expert_batch, agent_batch):
        return expert_batch, agent_batch

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if not args.drail_state_norm:
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
        if not self.args.drail_state_norm:
            return state
        state = state.cpu().numpy()

        if obsfilt is not None:
            state = obsfilt(state, update=False)
        state = torch.tensor(state).to(self.args.device)
        return state

    def _trans_agent_state(self, state, other_state=None):
        if not self.args.drail_state_norm:
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
        if self.args.disc_grad_pen != 0.0:
            grad_pen = self.args.disc_grad_pen * autils.wass_grad_pen(expert_states,
                    expert_actions, agent_states, agent_actions,
                    self.args.action_input, self._compute_disc_val)
            return grad_pen
        return 0

    def _compute_disc_val(self, state, action, label=None):
        label_one = self.discrim_net(state, action, 1.)
        label_zero = self.discrim_net(state, action, 0.)
        output = F.softmax(torch.stack([-label_one, -label_zero]),dim=0)[0]
        return output
    
    def plot_reward_map(self, i):
        x = torch.linspace(0, 10, 100)
        y = torch.linspace(-2, 2, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.reshape(-1, 1).to(self.args.device)
        Y = Y.reshape(-1,1).to(self.args.device)
        with torch.no_grad():
            s = self._compute_disc_val(X, Y)
            eps = 1e-20
            if self.args.reward_type == 'airl':
                reward = (s + eps).log() - (1 - s + eps).log()
            elif self.args.reward_type == 'gail':
                reward = (s + eps).log()
            elif self.args.reward_type == 'raw':
                reward = s
            elif self.args.reward_type == 'airl-positive':
                reward = (s + eps).log() - (1 - s + eps).log() + 20
            elif self.args.reward_type == 'revise':
                d_x = (s + eps).log()
                reward = d_x + (-1 - (-d_x).log())
            else:
                raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
            reward = reward.view(100, 100).cpu().numpy().T

        plt.figure(figsize=(8, 5))
        plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        file_path = "./data/imgs/" + self.args.prefix + "_reward_map.png"
        plt.savefig(file_path)
        return file_path
    
    def plot_disc_val_map(self, i):
        x = torch.linspace(0, 10, 100)
        y = torch.linspace(-2, 2, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.reshape(-1, 1).to(self.args.device)
        Y = Y.reshape(-1,1).to(self.args.device)
        with torch.no_grad():
            rewards = []
            for _ in range(10):
                reward = self._compute_disc_val(X, Y).view(100, 100).cpu().numpy().T
                rewards.append(reward)
            reward = torch.tensor(rewards).mean(dim=0)
        plt.figure(figsize=(8, 5))
        plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        file_path = "./data/imgs/" + self.args.prefix + "_disc_val_map.png"
        plt.savefig(file_path)
        return file_path

    def _compute_expert_loss(self, expert_d, expert_batch):
        return  F.binary_cross_entropy(expert_d,
                torch.ones(expert_d.shape).to(self.args.device))

    def _compute_agent_loss(self, agent_d, agent_batch):
        return  F.binary_cross_entropy(agent_d,
                torch.zeros(agent_d.shape).to(self.args.device))

    def _update_reward_func(self, storage, gradient_clip=False, t=1):
        self.discrim_net.train()

        log_vals = defaultdict(lambda: 0)
        obsfilt = self.get_env_ob_filt()

        expert_sampler, agent_sampler = self._get_sampler(storage)
        if agent_sampler is None:
            # algo requested not to update this step
            return {}
        n = 0
        for epoch_i in range(self.args.n_drail_epochs):
            for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
                expert_batch, agent_batch = self._trans_batches(
                    expert_batch, agent_batch)
                n += 1
                expert_d, agent_d, grad_pen = self._compute_discrim_loss(agent_batch, expert_batch, obsfilt)
                expert_loss = self._compute_expert_loss(expert_d, expert_batch)
                agent_loss = self._compute_agent_loss(agent_d, agent_batch)
                
                discrim_loss = expert_loss + agent_loss

                if self.args.disc_grad_pen != 0.0:
                    if t <= self.args.disc_grad_pen_period:
                        log_vals['grad_pen'] += grad_pen.item()
                        total_loss = discrim_loss + self.args.disc_grad_pen * grad_pen
                    else: 
                        log_vals['grad_pen'] += 0
                        total_loss = discrim_loss
                else:
                    total_loss = discrim_loss
                
                self.opt.zero_grad()
                total_loss.backward()
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.discrim_net.parameters(), max_norm=1.0)
                self.opt.step()

                log_vals['discrim_loss'] += discrim_loss.item()
                log_vals['expert_loss'] += expert_loss.item()
                log_vals['agent_loss'] += agent_loss.item()
                log_vals['expert_disc_val'] += expert_d.mean().item()
                log_vals['agent_disc_val'] += agent_d.mean().item()
                log_vals['agent_reward'] += ((agent_d + 1e-20).log() - (1 - agent_d + 1e-20).log()).mean().item()
                log_vals['dm_update_data'] += len(expert_batch)
                self.step += self.expert_train_loader.batch_size
        for k in log_vals:
            if k[0] != '_':
                log_vals[k] /= n
        if self.args.env_name[:4] == "Sine" and (self.step // (self.expert_train_loader.batch_size * n)) % 100 == 1 :
            # log_vals["_reward_map"] = self.plot_reward_map(self.step)
            log_vals["_disc_val_map"] = self.plot_disc_val_map(self.step)

        log_vals['dm_update_data'] *= n
        return log_vals
    
    def _compute_discrim_reward(self, storage, step, add_info):
        state = self._trans_agent_state(storage.get_obs(step))
        action = storage.actions[step]
        action = rutils.get_ac_repr(self.action_space, action)
            
        s = self._compute_disc_val(state, action)

        eps = 1e-20
        if self.args.reward_type == 'airl':
            reward = (s + eps).log() - (1 - s + eps).log()
        elif self.args.reward_type == 'gail':
            reward = (s + eps).log()
        elif self.args.reward_type == 'raw':
            reward = s
        elif self.args.reward_type == 'airl-positive':
            reward = (s + eps).log() - (1 - s + eps).log() + 20
        elif self.args.reward_type == 'revise':
            d_x = (s + eps).log()
            reward = d_x + (-1 - (-d_x).log())
        else:
            raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        return reward

    def _get_reward(self, step, storage, add_info):
        masks = storage.masks[step]
        with torch.no_grad():
            self.discrim_net.eval()
            reward = self._compute_discrim_reward(storage, step, add_info)

            if self.args.drail_reward_norm:
                if self.returns is None:
                    self.returns = reward.clone()

                self.returns = self.returns * masks * self.args.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), {}
            else:
                return reward, {}
    
    def eval_disc(self, state, action, label):
        self.discrim_net.eval()
        with torch.no_grad():
            discrim_output = self.discrim_net(state, action, label)
        return discrim_output

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides

        #########################################
        # New args
        #! TODO: Modify to fit Drail
        parser.add_argument('--action-input', type=str2bool, default=False)
        parser.add_argument('--drail-reward-norm', type=str2bool, default=False)
        parser.add_argument('--drail-state-norm', type=str2bool, default=True)
        parser.add_argument('--drail-action-norm', type=str2bool, default=False)
        parser.add_argument('--disc-lr', type=float, default=0.0001)
        parser.add_argument('--disc-grad-pen', type=float, default=0.0)
        parser.add_argument('--disc-grad-pen-period', type=float, default=1.0)
        parser.add_argument('--expert-loss-rate', type=float, default=1.0)
        parser.add_argument('--agent-loss-rate', type=float, default=-1.0)
        parser.add_argument('--agent-loss-rate-scheduler', type=str2bool, default=False)
        parser.add_argument('--agent-loss-end', type=float, default=-1.1)
        parser.add_argument('--discrim-depth', type=int, default=4)
        parser.add_argument('--discrim-num-unit', type=int, default=128)
        parser.add_argument('--n-drail-epochs', type=int, default=1)
        parser.add_argument('--label-dim', type=int, default=10)
        parser.add_argument('--test-sine-env', type=str2bool, default=False)
        parser.add_argument('--deeper-ddpm', type=str2bool, default=False)
        parser.add_argument('--reward-type', type=str, default='airl', help="""
                One of [Drail]. Changes the reward computation. Does
                not change training.
                """)

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        self.opt.load_state_dict(checkpointer.get_key('drail_disc_opt'))
        self.discrim_net.load_state_dict(checkpointer.get_key('drail_disc'))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key('drail_disc_opt', self.opt.state_dict())
        checkpointer.save_key('drail_disc', self.discrim_net.state_dict())
