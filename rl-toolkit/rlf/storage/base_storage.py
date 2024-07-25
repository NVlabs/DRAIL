from abc import abstractmethod

import rlf.rl.utils as rutils
import torch


class BaseStorage(object):
    def __init__(self):
        self._add_info_keys = []
        self.on_traj_done_fn = None

    def set_traj_done_callback(self, on_traj_done_fn):
        self._on_traj_done_callback = on_traj_done_fn

    def _on_traj_done(self, done_trajs):
        """
        done_trajs: A list of transitions where each transition is a tuple of form:
            (state,action,mask,info_dict,reward). The data is a bit confusing.
            mask[t] is technically the mask at t+1. The mask at t=0 is always
            1. The final state is NOT included and must be included through the
            info_dict if needed.
        """
        pass

    def init_storage(self, obs):
        self.traj_storage = [[] for _ in range(rutils.get_def_obs(obs).shape[0])]

    @abstractmethod
    def get_generator(self, **kwargs):
        pass

    def get_obs(self, step):
        pass

    def get_hidden_state(self, step):
        pass

    def get_masks(self, step):
        pass

    def insert(self, obs, next_obs, reward, done, info, ac_info):
        done_trajs = []

        for i in range(len(info)):
            traj_trans = self.get_traj_info(
                rutils.obs_select(obs, i),
                ac_info.take_action[i],
                done[i],
                info[i],
                reward[i],
            )
            self.traj_storage[i].append(traj_trans)

            if done[i]:
                done_trajs.append(self.traj_storage[i])
                self.traj_storage[i] = []
        if len(done_trajs) > 0:
            self._on_traj_done_callback(done_trajs)
            self._on_traj_done(done_trajs)

    def after_update(self):
        pass

    def to(self, device):
        pass

    def compute_masks(self, done, infos):
        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )

        return masks, bad_masks

    def get_traj_info(self, obs, action, done, info, reward):
        if done:
            mask = 0.0
        else:
            mask = 1.0

        ret_dict = {}
        for k in self.get_extract_info_keys():
            if k in info:
                assign_val = torch.tensor(info[k]).to(self.args.device)
                ret_dict[k] = assign_val

        return obs, action, mask, ret_dict, reward

    def add_info_key(self, key_name, data_size):
        """
        Defines a key from the info dictionary returned by the environment that
        should be stored.
        """
        self._add_info_keys.append(key_name)

    def get_extract_info_keys(self):
        return self._add_info_keys

    def get_add_info(self, key):
        raise NotImplemented("No add info is implemented for this storage type")
