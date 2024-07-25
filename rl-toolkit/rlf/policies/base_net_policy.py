import torch.nn as nn
import numpy as np
import torch
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.policies.base_policy import BasePolicy
from functools import partial
import inspect


class BaseNetPolicy(nn.Module, BasePolicy):
    """
    The starting point for all neural network policies. Includes an easy way to
    goal condition policies. Defines a base neural network transformation that
    outputs a hidden representation.
    """

    def __init__(self,
            use_goal=False,
            fuse_states=[],
            get_base_net_fn=None):
        """
        - get_base_fn: (ishape: tuple(int), recurrent: bool -> rlf.rl.model.BaseNet)
            returned module should take as input size of observation space and
            return the size `hidden_size`. NOTE if recurrent parameter is
            optional, if it is not specified, it will not be passed. The naming
            of the parameter has to be EXACTLY `recurrent`.
          default: none, use the default
        """
        super().__init__()
        if get_base_net_fn is None:
            get_base_net_fn = putils.get_img_encoder

        self.get_base_net_fn = get_base_net_fn
        self.fuse_states = fuse_states
        self.use_goal = use_goal

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        if 'recurrent' in inspect.getfullargspec(self.get_base_net_fn).args: #True
            self.get_base_net_fn = partial(self.get_base_net_fn,
                    recurrent=self.args.recurrent_policy)
        if self.use_goal: #False
            use_obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)
            if len(use_obs_shape) != 1:
                raise ValueError(('Goal conditioning only ',
                    'works with flat state representation'))
            use_obs_shape = (use_obs_shape[0] + obs_space['desired_goal'].shape[0],)
        else:
            use_obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)

        self.base_net = self.get_base_net_fn(use_obs_shape)
        base_out_dim = self.base_net.output_shape[0]
        for k in self.fuse_states:
            if len(obs_space.spaces[k].shape) != 1:
                raise ValueError('Can only fuse 1D states')
            base_out_dim += obs_space.spaces[k].shape[0]
        self.base_out_shape = (base_out_dim,)

    def _get_base_out_shape(self):
        return self.base_out_shape

    def _fuse_base_out(self, base_features, add_input):
        if len(self.fuse_states) == 0:
            return base_features
        fuse_states = torch.cat([add_input[k] for k in self.fuse_states], dim=-1)
        fused = torch.cat([base_features, fuse_states], dim=-1)
        return fused

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--recurrent-policy',
            action='store_true',
            default=False,
            help='use a recurrent policy')

    def get_storage_hidden_states(self):
        hxs = super().get_storage_hidden_states()
        if self.args.recurrent_policy:
            hxs['rnn_hxs'] = self.base_net.gru.hidden_size
        return hxs

    def _apply_base_net(self, state, add_state, hxs, masks):
        if self.use_goal:
            # Combine the goal and the state
            combined_state = torch.cat([state, add_state['desired_goal']], dim=-1)
            return self.base_net(combined_state, hxs, masks)
        else:
            return self.base_net(state, hxs, masks)

    def watch(self, logger):
        super().watch(logger)
        #logger.watch_model(self)
        print('Using policy network:')
        print(self)

    def save_to_checkpoint(self, checkpointer):
        checkpointer.save_key('policy', self.state_dict())
