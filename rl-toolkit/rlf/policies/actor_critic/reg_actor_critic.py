import numpy as np
import torch.nn as nn
import torch
import rlf.policies.utils as putils
from rlf.rl.model import def_mlp_weight_init
from rlf.policies.actor_critic.random_processes import *
from rlf.policies.actor_critic.base_actor_critic import ActorCritic
from rlf.policies.base_policy import create_simple_action_data
from functools import partial

NOISE_GENS = {
        'gaussian': GaussianNoise,
        'uh': UHNoise,
        }


class RegActorCritic(ActorCritic):
    """
    Defines an actor and critic type policy
    """

    def __init__(self,
            get_actor_fn=None,
            get_actor_head_fn=None,
            get_critic_fn=None,
            get_critic_head_fn=None,
            use_goal=False,
            fuse_states=[],
            get_base_net_fn=None):
        """
        - get_actor_fn:
          type: (obs_shape: tuple(int), input_shape: tuple(int) -> rlf.rl.model.BaseNet)
          Returned network should output (N,hidden_size) where hidden_size is
          arbitrary
        The actor output is scaled by the size of the action space.
        """

        if get_critic_fn is None:
            get_critic_fn = putils.get_def_ac_critic

        super().__init__(get_critic_fn, get_critic_head_fn, use_goal,
                fuse_states, get_base_net_fn)

        if get_actor_fn is None:
            get_actor_fn = putils.get_def_actor
        if get_actor_head_fn is None:
            get_actor_head_fn = putils.get_def_actor_head
        self.get_actor_fn = get_actor_fn
        self.get_actor_head_fn = get_actor_head_fn


    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

        # Can't work with discrete actions!
        assert self.action_space.__class__.__name__ == "Box"

        self.actor_net = self.get_actor_fn(obs_space.shape, self.base_net.output_shape)
        self.actor_head = self.get_actor_head_fn(self.actor_net.output_shape[0],
                self.action_space.shape[0])

        noise_gen_class = NOISE_GENS[self.args.noise_type]
        self.ac_low_bound = torch.tensor(self.action_space.low).to(args.device)
        self.ac_high_bound = torch.tensor(self.action_space.high).to(args.device)
        self.noise_gens = [noise_gen_class(size=self.action_space.shape,
                std=LinearSchedule(self.args.noise_std,
                    self.args.noise_decay_end,
                    self.args.noise_decay_step))
                for _ in range(self.args.num_processes)]
        if (self.action_space.high != -1.0 * self.action_space.low).any():
            raise ValueError("Asynmetric action space bounds currently not supported")


    def forward(self, state, add_state, hxs, masks):
        base_features, _ = self._apply_base_net(state, add_state, hxs, masks)
        actor_features, _ = self.actor_net(base_features, hxs, masks)
        return self.actor_head(actor_features) * self.action_space.high[0]

    def get_value(self, state, action, add_state, hxs, masks):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        critic_features, hxs = self.critic(base_features, action, hxs, masks)
        return self.critic_head(critic_features)

    def get_action(self, state, add_state, hxs, masks, step_info):
        should_resets = [True if m == 0.0 else False for m in masks]
        # Reset the noise for the beginning of every episode.
        for should_reset, noise_gen in zip(should_resets, self.noise_gens):
            if should_reset:
                noise_gen.reset_states()

        n_procs = state.shape[0]

        action = self.forward(state, add_state, hxs, masks)
        if not step_info.is_eval:
            cur_step = step_info.cur_num_steps
            if (cur_step >= self.args.n_rnd_steps) and (np.random.rand() >= self.args.rnd_prob):
                # Get the added noise.
                noise = torch.FloatTensor([ng.sample(cur_step)
                    for ng in self.noise_gens]).to(self.args.device)
                action += noise

                # Multi-dimensional clamp the action to the action space range.
                action = torch.min(torch.max(action, self.ac_low_bound), self.ac_high_bound)
            else:
                action = torch.tensor([self.action_space.sample()
                    for _ in range(n_procs)]).to(self.args.device)

        return create_simple_action_data(action, hxs)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--noise-type', type=str, default='gaussian',
                help='Noise type from [%s]' % ','.join(NOISE_GENS.keys()))
        parser.add_argument('--noise-std', type=float, default=0.2)
        parser.add_argument('--noise-decay-end', type=float, default=0.0)
        parser.add_argument('--noise-decay-step', type=int, default=-1,
                help=(
                    'Step to decay exploration noise to the specified ending',
                    ' value. Note that if -1 (default), no decay will occur.'
                    ))
        parser.add_argument('--n-rnd-steps', type=int, default=10000)
        parser.add_argument('--rnd-prob', type=float, default=0.0)

    def get_actor_params(self):
        return super().get_actor_params() + list(self.actor_head.parameters())

