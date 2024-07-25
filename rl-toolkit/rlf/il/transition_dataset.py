import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.il.il_dataset import ImitationLearningDataset, convert_to_tensors


class TransitionDataset(ImitationLearningDataset):
    """
    See `rlf/il/il_dataset.py` for notes about the demonstration dataset
    format.
    """

    def __init__(self, load_path, transform_dem_dataset_fn):
        super().__init__(load_path, transform_dem_dataset_fn)
        if load_path.endswith(".npz"):
            trajs = self._load_npz(load_path)
        else:
            trajs = self._load_pt(load_path)
        trajs = self._transform_dem_dataset_fn(trajs, trajs)
        trajs = convert_to_tensors(trajs)

        # Convert all to floats
        self.trajs = {}
        self.trajs["obs"] = trajs["obs"].float()
        self.trajs["done"] = trajs["done"].float()
        self.trajs["actions"] = trajs["actions"].float()
        self.trajs["next_obs"] = trajs["next_obs"].float()

        self.state_mean = torch.mean(self.trajs["obs"], dim=0)
        self.state_std = torch.std(self.trajs["obs"], dim=0)
        self._compute_action_stats()

    def _load_npz(self, load_path):
        x = np.load(load_path)
        n_trajs, traj_len, _ = x["obs"].shape
        N = n_trajs * traj_len
        done = torch.zeros(N)
        for i in range(n_trajs):
            done[(i + 1) * traj_len - 1] = 1

        obs = x["obs"].reshape(N, -1)
        this_obs = obs[:-1]
        next_obs = obs[1:]
        return {
            "obs": torch.tensor(this_obs),
            "rewards": torch.tensor(x["rews"].reshape(N, -1))[:-1],
            "actions": torch.tensor(x["acs"].reshape(N, -1))[:-1],
            # 1 if the transition resulted in termination
            "done": done[1:],
            "next_obs": torch.tensor(next_obs),
        }

    def _load_pt(self, load_path):
        return torch.load(load_path)

    def get_num_trajs(self):
        return int((self.trajs["done"] == 1).sum())

    def _compute_action_stats(self):
        self.action_mean = torch.mean(self.trajs["actions"], dim=0)
        self.action_std = torch.std(self.trajs["actions"], dim=0)

    def clip_actions(self, low_val, high_val):
        self.trajs["actions"] = rutils.multi_dim_clip(
            self.trajs["actions"], low_val, high_val
        )
        self._compute_action_stats()

    def to(self, device):
        for k in self.trajs:
            self.trajs[k] = self.trajs[k].to(device)
        return self

    def get_expert_stats(self, device):
        return {
            "state": (self.state_mean.to(device), self.state_std.to(device)),
            "action": (self.action_mean.to(device), self.action_std.to(device)),
        }

    def __len__(self):
        return len(self.trajs["obs"])

    def __getitem__(self, i):
        return {
            "state": self.trajs["obs"][i],
            "next_state": self.trajs["next_obs"][i],
            "done": self.trajs["done"][i],
            "actions": self.trajs["actions"][i],
        }

    def compute_split(self, traj_frac, rnd_seed=31):
        use_count = int(len(self) * traj_frac)
        all_idxs = np.arange(0, len(self))

        rng = np.random.default_rng(rnd_seed)
        rng.shuffle(all_idxs)
        idxs = all_idxs[:use_count]
        return torch.utils.data.Subset(self, idxs)
