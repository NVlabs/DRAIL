from rlf.policies.base_net_policy import BaseNetPolicy
import torch.nn.functional as F
import random
import math
import torch.nn as nn
import torch
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.policies.base_policy import create_simple_action_data


class SVGDPolicy(BaseNetPolicy):
    """
    Implementation of https://arxiv.org/pdf/1702.08165.pdf
    """
    def __init__(self, get_Q_base_fn=None,
            use_goal=False,
            fuse_states=[],
            get_base_net_fn=None):
        super().__init__(use_goal, fuse_states, get_base_net_fn)

        if get_Q_base_fn is None:
            get_Q_base_fn = putils.get_def_actor_head

        self.get_Q_base_fn = get_Q_base_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        if not rutils.is_discrete(action_space):
            raise ValueError("""
                    Only supports discrete actions. Continuous actions is much
                    harder in soft Q-learning.
                    """)
        # Create networks.
        self.softQ = self.get_Q_base_fn(self.base_net.output_shape[0],
                action_space.n)

    def forward(self, state, add_state=None, hxs=None, masks=None):
        base_features, _ = self._apply_base_net(state, add_state, hxs, masks)
        Q = self.softQ(base_features)

        # Directly from Eq (5) for computing the value function
        V = self.args.alpha * torch.log(torch.sum(torch.exp(Q / self.args.alpha), dim=1))
        V = V.unsqueeze(1)

        # Directly from Eq (6) for computing the policy
        pi = torch.exp((Q - V) / self.args.alpha)
        return pi

    def get_action(self, state, add_state, hxs, masks, step_info):
        dist = self.forward(state, add_state, hxs, masks)
        if step_info.is_eval:
            action = torch.argmax(dist, dim=1)
        else:
            action = torch.distributions.Categorical(dist).sample()
        return create_simple_action_data(action, hxs)

