from rlf.policies.base_net_policy import BaseNetPolicy
import torch.nn.functional as F
import random
import math
import torch.nn as nn
import torch
import rlf.policies.utils as putils
from rlf.policies.base_policy import create_simple_action_data


class DQN(BaseNetPolicy):
    """
    Defines approximation of Q(s,a) using deep neural networks. The value for
    each action are output as heads of the network.
    """

    def __init__(self,
            get_actor=None,
            use_goal=False,
            fuse_states=[],
            get_base_net_fn=None):
        super().__init__(use_goal, fuse_states, get_base_net_fn)
        if get_actor is None:
            get_actor = putils.get_def_actor_head
        self.get_actor = get_actor

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        assert self.action_space.__class__.__name__ == 'Discrete'
        self.actor = self.get_actor(self.base_net.output_shape[0],
                action_space.n)

    def forward(self, state, add_state=None, hxs=None, masks=None):
        base_features, _ = self._apply_base_net(state, add_state, hxs, masks)
        return self.actor(base_features)

    def get_action(self, state, add_state, hxs, masks, step_info):
        if step_info.is_eval:
            eps_threshold = 0
        else:
            num_steps = step_info.cur_num_steps
            eps_threshold = self.args.eps_end + \
                (self.args.eps_start - self.args.eps_end) * \
                math.exp(-1.0 * num_steps / self.args.eps_decay)

        sample = random.random()
        if sample > eps_threshold:
            q_vals = self.forward(state, add_state, hxs, masks)
            ret_action = q_vals.max(1)[1].unsqueeze(-1)
        else:
            # Take a random action.
            ret_action = torch.LongTensor([[random.randrange(self.action_space.n)]
                for i in range(state.shape[0])]).to(self.args.device)

        return create_simple_action_data(ret_action, hxs, {
            'alg_add_eps': eps_threshold
            })

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--eps-start', type=float, default=0.9)
        parser.add_argument('--eps-end', type=float, default=0.05)
        parser.add_argument('--eps-decay', type=float, default=200)
