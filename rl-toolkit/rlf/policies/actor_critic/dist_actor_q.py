"""
Code is heavily based off of https://github.com/denisyarats/pytorch_sac.
The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
"""
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
import torch
import torch.nn as nn
from rlf.policies.actor_critic.base_actor_critic import ActorCritic
from rlf.policies.base_net_policy import BaseNetPolicy
from rlf.policies.base_policy import create_simple_action_data
from rlf.rl.distributions import DiagGaussianActor
from rlf.rl.model import DoubleQCritic, MLPBase
import numpy as np


def get_sac_critic(obs_shape, in_shape, action_space, hidden_dim, depth=2):
    return DoubleQCritic(in_shape[0], action_space.shape[0], hidden_dim, depth)


def get_sac_actor(
    obs_shape, i_shape, action_space, log_std_bounds, hidden_dim, depth=2
):
    return DiagGaussianActor(
        i_shape[0], action_space.shape[0], hidden_dim, depth, log_std_bounds
    )


class DistActorQ(BaseNetPolicy):
    def __init__(
        self,
        get_actor_fn=None,
        get_critic_fn=None,
        use_goal=False,
        fuse_states=[],
        get_base_net_fn=None,
    ):

        super().__init__(use_goal, fuse_states, get_base_net_fn)

        if get_critic_fn is None:
            get_critic_fn = get_sac_critic
        self.get_critic_fn = get_critic_fn

        if get_actor_fn is None:
            get_actor_fn = get_sac_actor
        self.get_actor_fn = get_actor_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

        obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)

        self.critic = self.get_critic_fn(
            obs_shape=obs_shape,
            in_shape=self._get_base_out_shape(),
            action_space=action_space,
            hidden_dim=self.args.dist_q_hidden_dim,
        )

        log_std_bounds = [float(x) for x in self.args.log_std_bounds.split(",")]

        self.actor = self.get_actor_fn(
            obs_shape=rutils.get_obs_shape(obs_space, args.policy_ob_key),
            i_shape=self._get_base_out_shape(),
            action_space=action_space,
            log_std_bounds=log_std_bounds,
            hidden_dim=self.args.dist_q_hidden_dim,
        )

        self.ac_low_bound = torch.tensor(self.action_space.low).to(args.device).min()
        self.ac_high_bound = torch.tensor(self.action_space.high).to(args.device).max()

    def forward(self, state, add_state, hxs, masks):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        base_features = self._fuse_base_out(base_features, add_state)
        # dist = self.actor(base_features, hxs, masks)
        # try: 
        dist = self.actor(base_features, hxs, masks)
        # except:
            # breakpoint()
            # dist = self.actor(base_features, hxs, masks)

        return dist

    def get_action(self, state, add_state, hxs, masks, step_info):
        n_procs = state.shape[0]
        cur_step = step_info.cur_num_steps

        if not step_info.is_eval and cur_step < self.args.n_rnd_steps:
            sampled_actions = [self.action_space.sample() for _ in range(n_procs)]
            action_array = np.array(sampled_actions)
            action = torch.tensor(action_array).to(self.args.device)
            # action = torch.tensor(
            #     [self.action_space.sample() for _ in range(n_procs)]
            # ).to(self.args.device)
            return create_simple_action_data(action, hxs)

        dist = self.forward(state, add_state, hxs, masks)
        if step_info.is_eval:
            action = dist.mean
        else:
            action = dist.sample()
        action = action.clamp(self.ac_low_bound, self.ac_high_bound)

        return create_simple_action_data(action, hxs)

    def get_value(self, state, action, add_state, hxs, masks):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        return self.critic(base_features, action)

    def getV(self, state, alg):
        #import ipdb; ipdb.set_trace()
        dist = self.forward(state, None, None, None)
        n_action = dist.rsample()
        log_prob = dist.log_prob(n_action).sum(-1, keepdim=True)

        current_Q = self.critic(state, n_action)[0]
        current_V = current_Q - alg.alpha.detach() * log_prob
        
        return n_action, current_V

    def get_actor_params(self):
        return list(self.base_net.parameters()) + list(self.actor.parameters())

    def get_critic_params(self):
        return list(self.base_net.parameters()) + list(self.critic.parameters())

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--n-rnd-steps", type=int, default=10000)
        parser.add_argument("--log-std-bounds", type=str, default="-5,2")
        parser.add_argument(
            "--dist-q-hidden-dim",
            type=int,
            default=128,
            help="""The
        neural network hidden dimension for the actor and critic.
        """,
        )
