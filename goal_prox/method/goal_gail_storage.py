from rlf.storage.base_storage import BaseStorage
import random
import torch
import rlf.rl.utils as rutils
from collections import defaultdict
import numpy as np


class GoalGAILStorage(BaseStorage):
    def __init__(
        self, obs_space, action_space, capacity, relabel_ob_fn, is_reached_fn, args
    ):
        super().__init__()
        self.args = args
        self.capacity = capacity
        self.last_seen = None
        self.set_device = None
        self.d = self.args.device
        self.n_trajs = 0
        self.n_transitions = 0
        self.trajs = []
        self.on_policy_trajs = []
        self.future_p = 0.8

        self.relabel_ob = relabel_ob_fn
        self.is_reached = is_reached_fn

        # dummy for irl wrapper
        self.rewards = [0] * capacity

        obs_shape = obs_space.shape
        if rutils.is_discrete(action_space):
            self.action_dtype = torch.long
        else:
            self.action_dtype = torch.float32

        self.ob_keys = rutils.get_ob_shapes(obs_space)

    def _on_traj_done(self, done_trajs):
        for done_traj in done_trajs:
            self.n_transitions += len(done_traj)
            while self.n_transitions > self.capacity:
                self.n_transitions -= len(self.trajs[0])
                self.trajs.pop(0)
            self.trajs.append(done_traj)
            self.n_trajs = len(self.trajs)
            if len(self.on_policy_trajs) > 5:
                self.on_policy_trajs.pop(0)
            self.on_policy_trajs.append(done_traj)

    def insert(self, obs, next_obs, reward, done, infos, ac_info):
        done_trajs = []
        masks = []

        for i in range(len(infos)):
            mask = 0.0 if done[i] else 1.0
            # if infos[i]["ep_found_goal"] == 0.0:
            #     mask = 1.0
            if "bad_transition" in infos[i].keys() and self.args.use_proper_time_limits:
                mask = 1.0
            masks.append(mask)
            traj_trans = {
                "ob": rutils.get_def_obs(rutils.obs_select(obs, i)),
                "next_ob": rutils.get_def_obs(rutils.obs_select(next_obs, i)),
                "action": ac_info.take_action[i],
                "mask": mask,
            }
            self.traj_storage[i].append(traj_trans)

            if done[i]:
                dtype = self.traj_storage[i][-1]["next_ob"].dtype
                self.traj_storage[i][-1]["next_ob"] = torch.tensor(
                    infos[i]["final_obs"], dtype=dtype
                )
                done_trajs.append(self.traj_storage[i])
                self.traj_storage[i] = []

        if len(done_trajs) > 0:
            self._on_traj_done_callback(done_trajs)
            self._on_traj_done(done_trajs)

        self.last_seen = {
            "obs": next_obs,
            "masks": masks,
            "hxs": ac_info.hxs,
        }

    def sample_tensors(self, sample_size):
        return self._sample_tensors(self.trajs, sample_size)

    def _sample_tensors(self, trajs, sample_size):
        n_trajs = len(trajs)

        traj_idxs = np.random.randint(0, n_trajs, size=sample_size)
        trans_idxs = [
            np.random.randint(0, len(trajs[traj_idx])) for traj_idx in traj_idxs
        ]
        obs = []
        next_obs = []
        actions = []
        rewards = []
        masks = []
        for i in range(sample_size):
            trans = trajs[traj_idxs[i]][trans_idxs[i]]
            ob = trans["ob"]
            next_ob = trans["next_ob"]
            mask = trans["mask"]
            action = trans["action"]

            # HER future sampling
            replace_goal = np.random.uniform() < self.future_p
            if replace_goal:
                future_idx = np.random.randint(trans_idxs[i], len(trajs[traj_idxs[i]]))
                future_ob = trajs[traj_idxs[i]][future_idx]["next_ob"]
                ob = self.relabel_ob(ob, future_ob)
                next_ob = self.relabel_ob(next_ob, future_ob)

            success = self.is_reached(next_ob)

            obs.append(ob)
            next_obs.append(next_ob)
            actions.append(action)
            rewards.append(success)
            masks.append(mask)

        obs = torch.stack(obs).to(device=self.d)
        next_obs = torch.stack(next_obs).to(device=self.d)
        actions = torch.stack(actions).to(device=self.d)
        rewards = torch.tensor(rewards, device=self.d, dtype=torch.float32)
        masks = torch.tensor(masks, device=self.d, dtype=torch.float32)

        # Algorithms expect 2D tensor for rewards and masks.
        rewards = rewards.view(rewards.shape[0], 1)
        masks = masks.view(masks.shape[0], 1)

        cur_add = {
            "masks": None,
            "add_state": {},
            "hxs": {},
        }
        next_add = {
            "masks": masks,
            "add_state": {},
            "hxs": {},
        }

        return obs, next_obs, actions, rewards, cur_add, next_add

    def __len__(self):
        return self.n_trajs

    def init_storage(self, obs):
        super().init_storage(obs)
        batch_size = rutils.get_def_obs(obs).shape[0]
        hxs = {}
        self.last_seen = {
            "obs": obs,
            "masks": torch.zeros(batch_size, 1),
            "hxs": hxs,
        }

    def get_obs(self, step):
        ret_obs = self.last_seen["obs"]
        return ret_obs

    def get_hidden_state(self, step):
        return self.last_seen["hxs"]

    def get_masks(self, step):
        return self.last_seen["masks"]

    def to(self, device):
        self.d = device
        self.set_device = device

    def get_generator(self, num_batch, batch_size):
        if self.n_trajs == 0:
            return
        for _ in range(num_batch):
            batch = self._sample_tensors(self.on_policy_trajs, batch_size)
            obs, next_obs = batch[:2]
            mask = batch[-1]["masks"]
            yield {
                "state": obs,
                "next_state": next_obs,
                "mask": mask,
            }

    def get_add_info(self, key):
        return {}

    def set_reward_function(self, reward_fn):
        self.reward_fn = reward_fn
