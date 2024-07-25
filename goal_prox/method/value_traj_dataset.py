import numpy as np
from goal_prox.method.goal_traj_dataset import GoalTrajDataset
import torch
from goal_prox.envs.gw_helper import *

def exp_discounted(T, t, delta):
    return np.power(delta, T - t)

def linear_discounted(T, t, delta):
    return max(1.0 - ((T - t) * delta), 0.0)

def big_discounted(T, t, delta, start_val):
    return max(start_val - ((T - t) * delta), 0.0)

def compute_discounted_prox(T, compute_prox_fn):
    return np.array([compute_prox_fn(T, t + 1) for t in range(T)],
            dtype=np.float32)


class ValueTrajDataset(GoalTrajDataset):
    def get_prox_stats(self):
        proxs = [x[1] for x in self.data]
        return np.min(proxs), np.max(proxs)

    def __init__(self, load_path, compute_prox_fn, args):
        self.compute_prox_fn = compute_prox_fn
        self.args = args
        super().__init__(load_path)

    def _gen_data(self, trajs):
        data = []
        for states, actions in trajs:
            T = len(states)
            proxs = torch.tensor(compute_discounted_prox(T, self.compute_prox_fn))
            # The last action is all 0.
            use_actions = torch.cat([actions, torch.zeros(1, *actions.shape[1:])], dim=0)
            data.extend(zip(states, proxs, use_actions))
        return data

    def __getitem__(self, i):
        return {
                'state': self.data[i][0],
                'prox': self.data[i][1],
                'actions': self.data[i][2]
                }


