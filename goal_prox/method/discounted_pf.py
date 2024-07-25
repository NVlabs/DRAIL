from functools import partial

import rlf.rl.utils as rutils
import torch
import torch.nn.functional as F
from goal_prox.method.prox_func import ProxFunc
from goal_prox.method.value_traj_dataset import *
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO


class DiscountedProxIL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_pf_base=None):
        super().__init__(
            [DiscountedProxFunc(get_pf_base=get_pf_base), agent_updater],
            designated_rl_idx=1,
        )


class DiscountedProxFunc(ProxFunc):
    def _prox_func_iter(self, data_batch):
        states = data_batch["state"].to(self.args.device)
        proximity = data_batch["prox"].to(self.args.device)
        actions = data_batch["actions"].to(self.args.device)

        guess_proximity = self._get_prox_vals(states, actions)

        n_ensembles = guess_proximity.shape[0]
        loss = F.mse_loss(
            guess_proximity.view(n_ensembles, -1),
            proximity.view(1, -1).repeat(n_ensembles, 1),
        )
        return loss

    def _get_prox_val_fn(self):
        use_fn = None
        if self.args.dmode == "exp":
            use_fn = exp_discounted
            def_delta = 0.95
        elif self.args.dmode == "linear":
            use_fn = linear_discounted
            def_delta = 0.001
        elif self.args.dmode == "big":
            use_fn = partial(big_discounted, start_val=self.args.start_delta)
            def_delta = 0.001
        else:
            raise ValueError("Must specify discounting mode")

        if self.args.pf_delta is None:
            self.args.pf_delta = def_delta

        return partial(use_fn, delta=self.args.pf_delta)

    def _get_traj_dataset(self, traj_load_path):
        self.compute_prox_fn = self._get_prox_val_fn()
        return ValueTrajDataset(traj_load_path, self.compute_prox_fn, self.args)

    def compute_good_traj_prox(self, obs, actions):
        return torch.tensor(compute_discounted_prox(len(obs), self.compute_prox_fn))

    def compute_bad_traj_prox(self, obs, actions):
        return torch.zeros(len(obs))

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # New args
        parser.add_argument("--dmode", type=str, default="exp")
        parser.add_argument("--start-delta", type=float, default=1.0)
        parser.add_argument("--pf-delta", type=float, default=0.95)
