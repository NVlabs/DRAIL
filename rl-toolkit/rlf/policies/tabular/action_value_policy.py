from rlf.policies.base_policy import BasePolicy
from rlf.policies.base_policy import create_np_action_data
import torch
import random
import numpy as np

class ActionValuePolicy(BasePolicy):
    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.Q = np.array([args.q_init for _ in range(action_space.n)])

    def get_action(self, state, add_state, hxs, masks, step_info):
        if random.random() > self.args.eps:
            a = np.argmax(self.Q)
        else:
            a = self.action_space.sample()
        return create_np_action_data(a)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--q-init', type=float, default=0.0)
        parser.add_argument('--eps', type=float, default=0.01)
