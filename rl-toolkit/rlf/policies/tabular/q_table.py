from rlf.policies.base_policy import BasePolicy, create_np_action_data
import numpy as np
import math
import random
import torch
from collections import defaultdict

class QTable(BasePolicy):
    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.Q = defaultdict(lambda: np.zeros(action_space.n))
        if self.args.eps_end == None:
            self.args.eps_end = self.args.eps_start

    def get_action_probs(self, state, step_info):
        if isinstance(state, int):
            q_s = self.Q[state]
        else:
            if state.shape[-1] == 1:
                q_s = self.Q[state[0].long().item()]
            else:
                q_s = self.Q[tuple(state[0].numpy())]

        if step_info.is_eval:
            add_eps = 0
        else:
            num_steps = step_info.cur_num_steps
            add_eps = self.args.eps_end + \
                (self.args.eps_start - self.args.eps_end) * \
                math.exp(-1.0 * num_steps / self.args.eps_decay)
        self.add_eps = add_eps

        nA = len(q_s)
        sel_probs = np.ones(nA, dtype=float) * add_eps / nA
        best_action = np.random.choice(np.flatnonzero(q_s == q_s.max()))
        sel_probs[best_action] += (1.0 - add_eps)
        return sel_probs

    def get_action(self, state, add_state=None, hxs=None, masks=None, step_info=None):
        sel_probs = self.get_action_probs(state, step_info)
        ret_action = np.random.choice(np.arange(len(sel_probs)), p=sel_probs)

        return create_np_action_data(ret_action, {
            'alg_add_eps': self.add_eps
            })

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--eps-start', type=float, default=0.1)
        parser.add_argument('--eps-end', type=float, default=None,
            help="""
            If "None" then no decay is applied.
            """)
        parser.add_argument('--eps-decay', type=float, default=200)
