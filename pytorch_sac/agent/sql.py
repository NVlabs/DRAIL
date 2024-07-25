import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent import Agent
import utils

import hydra
import rlf.algos.utils as autils
from torch.distributions import Categorical


class SQLAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, obs_shape, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, linear_lr_decay, trunk_type):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.linear_lr_decay = linear_lr_decay

        # optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)
        self.alpha = 1 / init_temperature

        self.critic_lr = critic_lr

        self.train()
        self.critic_target.train()

    def set_config(self, config):
        self.config = config

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        Q = self.critic(obs)
        V = self.compute_value(Q).squeeze()
        dist = torch.exp((Q - V) / self.alpha)
        action = Categorical(dist).sample()
        return action.cpu().item()


    def compute_value(self, q_vals):
        return self.alpha * torch.logsumexp(q_vals / self.alpha, dim=1, keepdim=True)

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):
        next_Q = self.critic_target(next_obs)
        next_V = self.compute_value(next_Q)
        target_Q = reward + (not_done * self.discount * next_V)
        target_Q = target_Q.detach()

        current_Q = self.critic(obs).gather(1, action.long())
        critic_loss = F.mse_loss(current_Q, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)
        total_steps = self.config.num_train_steps

        if self.linear_lr_decay:
            # The critic is updated every step.
            autils.linear_lr_schedule(step, total_steps, self.critic_lr,
                    self.critic_optimizer)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                           logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
