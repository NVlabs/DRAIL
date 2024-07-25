from rlf.policies.base_policy import BasePolicy, create_np_action_data
import numpy as np


class TabularPolicy(BasePolicy):
    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        # Random init.
        self.pi = [np.random.randint(action_space.n) for _ in range(obs_space.n)]

    def get_action(self, state, add_state, hxs, masks, step_info):
        state = state[0].long().item()
        action = self.pi[state]
        return create_np_action_data(action)
