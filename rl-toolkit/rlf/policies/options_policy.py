from rlf.policies.base_net_policy import BaseNetPolicy
import rlf.policies.utils as putils
from torch.distributions import Categorical, Bernoulli
import torch.nn as nn
from rlf.policies.base_policy import ActionData
import gym.spaces as spaces
import math
import numpy as np
import torch

class OptionsPolicy(BaseNetPolicy):
    def __init__(self,
            get_critic_fn=None,
            get_term_fn=None,
            get_option_fn=None,
            use_goal=False,
            get_base_net_fn=None):
        # We don't want any base encoder for non-image envs so we can separate
        # the neural networks for the intra-option, option and termination
        # policies. We can use a shared image encoding when working with images
        if get_base_net_fn is None:
            get_base_net_fn = putils.get_img_encoder

        if get_critic_fn is None:
            get_critic_fn = putils.get_mlp_net_var_out_fn((64, 64))
        if get_term_fn is None:
            get_term_fn = putils.get_mlp_net_var_out_fn((64, 64))
        if get_option_fn is None:
            get_option_fn = putils.get_mlp_net_var_out_fn((64, 64))

        super().__init__(use_goal, get_base_net_fn)
        self.get_critic_fn = get_critic_fn
        self.get_term_fn = get_term_fn
        self.get_option_fn = get_option_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        in_shape = self.base_net.output_shape

        if not isinstance(action_space, spaces.Discrete):
            raise ValueError(("Currently option critic only supports discrete",
                    "action space environments. The change to make it work with",
                    "continuous action space envrionments is actually very",
                    "easy, so it won't be hard to add."))

        self.critic_net = self.get_critic_fn(in_shape, self.args.n_options)
        self.term_net = self.get_term_fn(in_shape, self.args.n_options)

        self.option_nets = nn.ModuleList([
            self.get_option_fn(in_shape, action_space.n)
            for _ in range(self.args.n_options)
            ])

    def _sel_option(self, state, add_state, hxs, masks, eps_threshold):
        if np.random.rand() < eps_threshold:
            return torch.LongTensor([np.random.choice(self.args.n_options)
                for i in range(state.shape[0])])
        else:
            values = self.get_value(state, hxs, masks)
            return values.argmax(-1)

    def get_value(self, state, hxs, masks):
        return self.critic_net(state, hxs, masks)[0]

    def get_term_prob(self, state, hxs, masks):
        term_logits, _ = self.term_net(state, hxs, masks)
        term_logits = term_logits.sigmoid()
        return term_logits

    def get_action(self, state, add_state, hxs, masks, step_info):
        num_steps = step_info.cur_num_steps
        eps_threshold = self.args.eps_end + \
                (self.args.eps_start - self.args.eps_end) * \
                math.exp(-1.0 * num_steps / self.args.eps_decay)

        prev_option = hxs['option'].long()
        term = hxs['term']

        # If the mask is 0, the episode is over and we must decide a new
        # option.
        new_option_sel = self._sel_option(state, add_state, hxs, masks,
                eps_threshold)
        new_option_sel = new_option_sel.unsqueeze(-1)

        batch_size = len(masks)
        use_options = torch.zeros(batch_size,1).long()

        for i in range(batch_size):
            if masks[i] == 0 or term[i] == 1.0:
                use_option = new_option_sel[i]
            else:
                use_option = prev_option[i]
            use_options[i] = use_option

        action, action_log_probs, dist_entropy = self.get_actions(state, hxs,
                masks, use_options)

        term_logits = self.get_term_prob(state, hxs, masks)
        sel_logits = term_logits.gather(1, use_options)
        term = Bernoulli(sel_logits).sample()

        hxs = {
                'option': use_options,
                'term': term,
                }

        return ActionData(torch.zeros(batch_size,1), action, action_log_probs, hxs, {
            'alg_add_termination': term.mean().item(),
            'alg_add_termination_std': term.std().item(),
            'alg_add_option': use_options.float().mean().item(),
            'alg_add_option_std': use_options.float().mean().std(),
            })

    def get_actions(self, state, hxs, masks, use_options):
        logits = []
        for i in range(state.shape[0]):
            logits.append(
                    self.option_nets[use_options[i]](state[i], hxs, masks)[0]
                    )
        logits = torch.stack(logits)
        probs = (logits / self.args.temp).softmax(dim=-1)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy()
        return action, action_log_probs, dist_entropy

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--n-options", type=int, default=3,
            help="Number of option sub-policies")
        parser.add_argument('--eps-start', type=float, default=1.0)
        parser.add_argument('--eps-end', type=float, default=0.1)
        parser.add_argument('--eps-decay', type=float, default=20000)
        parser.add_argument('--temp', type=float, default=0.001)
