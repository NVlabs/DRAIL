from rlf.policies.base_policy import BasePolicy, create_simple_action_data
import torch
import rlf.rl.utils as rutils
from gym import spaces

class RandomPolicy(BasePolicy):
    def get_action(self, state, add_state, hxs, masks, step_info):
        n_procs = rutils.get_def_obs(state).shape[0]
        action = torch.tensor([self.action_space.sample()
            for _ in range(n_procs)]).to(self.args.device)
        if isinstance(self.action_space, spaces.Discrete):
            action = action.unsqueeze(-1)

        return create_simple_action_data(action, hxs)
