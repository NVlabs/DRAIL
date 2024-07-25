import torch

from rlf.algos.il.gaifo import GaifoDiscrim
from rlf.algos.nested_algo import NestedAlgo
from goal_prox.method.goal_gail_algo import GoalGAILAlgo
from goal_prox.method.goal_gail_dataset import GoalGAILTrajDataset


class GoalGAIL(NestedAlgo):
    def __init__(self, agent_updater=GoalGAILAlgo(), get_discrim=None):
        super().__init__([GoalGAILDiscrim(get_discrim), agent_updater], 1)


class GoalGAILDiscrim(GaifoDiscrim):
    def _get_traj_dataset(self, traj_load_path):
        return GoalGAILTrajDataset(traj_load_path, self.args)

    def _trans_batches(self, expert_batch, agent_batch):
        return expert_batch, agent_batch

    def _get_sampler(self, storage):
        expert_experience = self.expert_dataset.get_generator(
            num_batch=1,
            batch_size=self.args.traj_batch_size,
            relabel_ob=storage.relabel_ob,
            is_reached=storage.is_reached,
        )
        agent_experience = storage.get_generator(
            num_batch=1, batch_size=self.args.traj_batch_size
        )
        return expert_experience, agent_experience

    def update(self, storage):
        self.update_i += 1

        storage.set_reward_function(self._compute_reward)

        if len(storage) < 1:
            return {}

        if self.args.goal_gail_weight == 0:
            return {}

        if self.update_i % self.args.goal_gail_update_every:
            return {}

        log_vals = self._update_reward_func(storage)

        return log_vals

    def _compute_reward(self, obs, next_obs):
        state = obs
        next_state = next_obs

        d_val = self.discrim_net(state, next_state)
        s = torch.sigmoid(d_val)
        eps = 1e-20
        if self.args.reward_type == 'airl':
            reward = (s + eps).log() - (1 - s + eps).log()
        elif self.args.reward_type == 'gail':
            reward = (s + eps).log()
        elif self.args.reward_type == 'raw':
            reward = d_val
        elif self.args.reward_type == 'gaifo':
            reward = -1.0 * (s + eps).log()
        else:
            raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        return reward

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--goal-gail-update-every', type=int, default=10)
