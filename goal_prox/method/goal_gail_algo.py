from collections import defaultdict

import torch

import rlf.rl.utils as rutils
from rlf.algos.off_policy.sac import SAC
from rlf.algos.off_policy.ddpg import DDPG

from goal_prox.method.goal_gail_storage import GoalGAILStorage
import numpy as np


def create_storage_buff(
    obs_space, action_space, buff_size, relabel_ob_fn, is_reached_fn, args
):
    return GoalGAILStorage(
        obs_space, action_space, buff_size, relabel_ob_fn, is_reached_fn, args
    )


class GoalGAILAlgo(DDPG):
    def __init__(self, create_storage_buff_fn=create_storage_buff):
        super().__init__()
        self.create_storage_buff_fn = create_storage_buff_fn

    def init(self, policy, args):
        args.trans_buffer_size = int(args.trans_buffer_size)
        super().init(policy, args)

    def get_storage_buffer(self, policy, envs, args):
        return self.create_storage_buff_fn(
            policy.obs_space,
            policy.action_space,
            args.trans_buffer_size,
            envs.unwrapped.envs[0].relabel_ob,
            envs.unwrapped.envs[0].is_reached,
            args,
        )

    def _sample_transitions(self, storage):
        obs, next_obs, actions, rewards, cur_add, next_add = storage.sample_tensors(
            self.args.batch_size
        )
        rewards = rewards + self.args.goal_gail_weight * storage.reward_fn(obs, next_obs)
        return obs, next_obs, actions, rewards, cur_add, next_add

    def update(self, storage):
        super().update(storage)
        if len(storage) < 1:
            return {}

        ns = self.get_completed_update_steps(self.update_i)
        if ns % self.args.update_every != 0:
            return {}

        avg_log_vals = defaultdict(list)
        for i in range(self.args.updates_per_batch):
            log_vals = self._optimize(*self._sample_transitions(storage))
            for k, v in log_vals.items():
                avg_log_vals[k].append(v)

        avg_log_vals = {k: np.mean(v) for k, v in avg_log_vals.items()}

        return avg_log_vals

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--goal-gail-weight', type=float, default=0.1)
