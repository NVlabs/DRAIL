from rlf.policies.base_policy import BasePolicy, create_simple_action_data
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import defaultdict

class SolvePolicy(BasePolicy, ABC):
    """
    Plan all actions ahead of time based on the starting state. This can be
    useful for classical planning algorithms such as path planning algorithms.
    """

    def __init__(self):
        super().__init__()
        self.is_first = True

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.all_actions = defaultdict(list)
        self.ep_idx = defaultdict(lambda: 0)

    @abstractmethod
    def _solve_env(self, state):
        pass

    def get_action(self, state, hxs, masks, step_info):
        if self.is_first:
            masks = torch.zeros(masks.shape)

        sel_actions = []
        for i, mask in enumerate(masks):
            if mask == 0.0:
                if len(self.all_actions[i]) != 0:
                    # We should have exhaused all of the actions
                    assert self.ep_idx[i] == len(self.all_actions[i]) - 1
                # The env has reset, solve it.
                actions = self._solve_env(state[i])
                self.all_actions[i] = actions
                self.ep_idx[i] = 0
            else:
                self.ep_idx[i] += 1
            sel_actions.append(self.all_actions[i][self.ep_idx[i]])
        sel_actions = torch.tensor(sel_actions).unsqueeze(-1)
        self.is_first = False
        return create_simple_action_data(sel_actions)

