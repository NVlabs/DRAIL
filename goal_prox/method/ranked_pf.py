from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from goal_prox.method.prox_func import ProxFunc
import torch.nn as nn
import torch.nn.functional as F
import torch
from goal_prox.method.goal_traj_dataset import GoalTrajDataset
import torch
import numpy as np


class RankedProxIL(NestedAlgo):
    def __init__(self, agent_updater=PPO()):
        super().__init__(
            [RankedProxFunc(), agent_updater], designated_rl_idx=1)


def gen_rank_pairs(states, actions):
    """
    For now, I am putting a dummy value for the actions.
    """
    for i in range(len(states) - 1):
        yield states[i], actions[0], states[i+1], actions[0]

def gen_rank_pairs_from_expert(agent_states, expert_dataset):
    expert_idx = np.random.randint(0, len(expert_dataset), len(agent_states))
    for i, j in enumerate(expert_idx):
        yield {
            's0': agent_states[i],
            'a0': None,
            's1': expert_dataset[j]['s0'],
            'a1': None
        }


class RankedTrajDataset(GoalTrajDataset):
    def _gen_data(self, trajs):
        data = []
        for states, actions in trajs:
            # Get pairwise examples from this state trajectory.
            data.extend(gen_rank_pairs(states, actions))
        return data

    def __getitem__(self, i):
        return {
            's0': self.data[i][0],
            'a0': self.data[i][1],
            's1': self.data[i][2],
            'a1': self.data[i][3],
        }


class RankedProxFunc(ProxFunc):
    def _prox_func_iter(self, data_batch):
        s0 = data_batch['s0'].to(self.args.device)
        a0 = data_batch['a0'].to(self.args.device)

        s1 = data_batch['s1'].to(self.args.device)
        a1 = data_batch['a1'].to(self.args.device)

        s0_logits = self.prox_func(s0, a0)[0]
        s1_logits = self.prox_func(s1, a1)[0]
        logits = torch.cat([s0_logits, s1_logits], dim=-1)

        targets = torch.ones((len(logits),), dtype=torch.int64).to(self.args.device)
        loss = F.cross_entropy(logits, targets)

        return loss

    def _get_traj_dataset(self, traj_load_path):
        return RankedTrajDataset(traj_load_path)

    def _get_traj_tuples(self, actions, masks, raw_obs, final_state,
            end_t, was_success):
        traj_actions = actions[:end_t]
        traj_masks = masks[:end_t]
        traj_raw_obs = raw_obs[:end_t]
        traj_raw_obs = torch.cat([traj_raw_obs, final_state[:,end_t-1]])

        if not was_success:
            if self.args.below_expert:
                return list(gen_rank_pairs_from_expert(traj_raw_obs, self.expert_dataset))
            else:
                return []

        ranked = gen_rank_pairs(traj_raw_obs, actions)
        for state, action, n_state, n_action in ranked:
            yield {
                's0': state,
                'a0': action,
                's1': n_state,
                'a1': n_action,
            }

    def should_use_failure(self):
        return self.args.below_expert

    def init(self, policy, args):
        super().init(policy, args)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--below-expert', action='store_true')
