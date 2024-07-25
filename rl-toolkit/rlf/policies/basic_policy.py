from rlf.policies.base_net_policy import BaseNetPolicy
import torch.nn.functional as F
import random
import math
import gym
import torch.nn as nn
import torch
from rlf.policies.base_policy import create_simple_action_data
import rlf.policies.utils as putils
import rlf.rl.utils as rutils

class BasicPolicy(BaseNetPolicy):
    def __init__(self,
            is_stoch=False,
            fuse_states=[],
            use_goal=False,
            get_base_net_fn=None):
        super().__init__(use_goal, fuse_states, get_base_net_fn)
        self.state_norm_fn = lambda x: x
        self.action_denorm_fn = lambda x: x
        self.is_stoch = is_stoch

    def set_state_norm_fn(self, state_norm_fn):
        self.state_norm_fn = state_norm_fn

    def set_action_denorm_fn(self, action_denorm_fn):
        self.action_denorm_fn = action_denorm_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        ac_dim = rutils.get_ac_dim(action_space)
        self.action_head = nn.Linear(self.base_net.output_shape[0], ac_dim)
        if not rutils.is_discrete(self.action_space) and self.is_stoch:
            self.std = nn.Linear(self.base_net.output_shape[0], ac_dim)

    def forward(self, state, rnn_hxs, mask):
        # USES RSAMPLE
        base_features, _ = self.base_net(state, rnn_hxs, mask)
        ret_action = self.action_head(base_features)
        if not rutils.is_discrete(self.action_space) and self.is_stoch:
            std = self.std(base_features)
            dist = torch.distributions.Normal(ret_action, std)
            ret_action = dist.rsample()

        return ret_action, None, None

    def get_action(self, state, add_state, rnn_hxs, mask, step_info):
        base_features, _ = self.base_net(state, rnn_hxs, mask)

        ret_action = self.action_head(base_features)
        if step_info.is_eval or not self.is_stoch:
            ret_action = rutils.get_ac_compact(self.action_space, ret_action)
        else:
            if rutils.is_discrete(self.action_space):
                dist = torch.distributions.Categorical(ret_action.softmax(dim=-1))
            else:
                std = self.std(base_features)
                dist = torch.distributions.Normal(ret_action, std)
            ret_action = dist.sample()

        return create_simple_action_data(ret_action, rnn_hxs)
