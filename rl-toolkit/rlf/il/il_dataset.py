from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch
import torch.utils.data


def convert_to_tensors(trajs):
    if not isinstance(trajs["obs"], dict) and not isinstance(
        trajs["obs"], torch.Tensor
    ):
        trajs["obs"] = torch.tensor(trajs["obs"])
    if not isinstance(trajs["done"], torch.Tensor):
        trajs["done"] = torch.tensor(trajs["done"])
    if not isinstance(trajs["actions"], torch.Tensor):
        trajs["actions"] = torch.tensor(trajs["actions"])
    if not isinstance(trajs["next_obs"], dict) and not isinstance(
        trajs["next_obs"], torch.Tensor
    ):
        trajs["next_obs"] = torch.tensor(trajs["next_obs"])
    return trajs


class ImitationLearningDataset(torch.utils.data.Dataset, ABC):
    """
    The data should be a dictionary saved with `torch.save`, consisting of
        {
        'done': torch.tensor
        'obs': torch.tensor
        'next_obs': torch.tensor
        'actions': torch.tensor
        }
        All tensors should be exactly the same length.
    """

    def __init__(
        self, load_path, transform_dem_dataset_fn: Callable[[Dict, Dict], Dict] = None
    ):
        """
        :param transform_dem_dataset_fn: Takes as input the output trajectory
        and the original data trajectory and outputs the modified trajectory.
        """
        if transform_dem_dataset_fn is None:
            transform_dem_dataset_fn = lambda x, y: x
        self._transform_dem_dataset_fn = transform_dem_dataset_fn

    def viz(self, args):
        pass

    @abstractmethod
    def get_expert_stats(self, device):
        pass

    @abstractmethod
    def get_num_trajs(self):
        pass

    @abstractmethod
    def compute_split(self, traj_frac, rnd_seed):
        pass

    def clip_actions(self, low_val, high_val):
        pass

    def to(self, device):
        return self
