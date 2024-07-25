from collections import defaultdict

import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.storage.base_storage import BaseStorage
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def get_shape_for_ac(action_space):
    if action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    elif action_space.__class__.__name__ == "Dict":
        # Another example of being hardcoded to logic game.
        # 1 for the discrete action space index selection
        action_shape = 1 + action_space.spaces["pos"].shape[0]
    else:
        action_shape = action_space.shape[0]

    return action_shape


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


def to_double(inp):
    return inp.double() if (inp is not None and inp.dtype == torch.float32) else inp


class RolloutStorage(BaseStorage):
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_space,
        action_space,
        args,
        value_dim=1,
        hidden_states={},
    ):
        """
        - hidden_states: dict(key_name: str -> hidden_state_dim: int)
        """
        super().__init__()
        self.value_dim = value_dim
        self.args = args

        self.ob_keys = rutils.get_ob_shapes(obs_space)
        self.obs = {}
        for k, space in self.ob_keys.items():
            ob = torch.zeros(num_steps + 1, num_processes, *space)
            if k is None:
                self.obs = ob
            else:
                self.obs[k] = ob

        # Data from the info dictionary that will be saved.
        self.add_data = {}

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, self.value_dim)
        self.returns = torch.zeros(num_steps + 1, num_processes, self.value_dim)
        self.action_log_probs = torch.zeros(num_steps, num_processes, self.value_dim)

        self.hidden_states = {}
        for k, dim in hidden_states.items():
            self.hidden_states[k] = torch.zeros(num_steps + 1, num_processes, dim)

        action_shape = get_shape_for_ac(action_space)

        self.actions = torch.zeros(num_steps, num_processes, action_shape)

        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()

        self.masks = torch.zeros(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.n_procs = num_processes
        self.step = 0

    def get_def_obs_seq(self):
        if isinstance(self.obs, dict):
            return rutils.get_def_obs(self.obs)
        else:
            return self.obs

    def add_info_key(self, key_name, data_size):
        super().add_info_key(key_name, data_size)
        self.add_data[key_name] = torch.zeros(self.num_steps, self.n_procs, *data_size)

    def init_storage(self, obs):
        super().init_storage(obs)
        for k in self.ob_keys:
            if k is None:
                self.obs[0].copy_(obs)
            else:
                self.obs[k][0].copy_(obs[k])

    def get_add_info(self, key):
        return self.add_data[key]

    def to(self, device):
        for k in self.ob_keys:
            if k is None:
                self.obs = self.obs.to(device)
            else:
                self.obs[k] = self.obs[k].to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        for k, d in self.add_data.items():
            self.add_data[k] = d.to(device)

        for k, d in self.hidden_states.items():
            self.hidden_states[k] = d.to(device)

    def insert(self, obs, next_obs, rewards, done, info, ac_info):
        super().insert(obs, next_obs, rewards, done, info, ac_info)
        masks, bad_masks = self.compute_masks(done, info)

        for k in self.ob_keys:
            if k is None:
                self.obs[self.step + 1].copy_(next_obs)
            else:
                self.obs[k][self.step + 1].copy_(next_obs[k])

        for i, inf in enumerate(info):
            for k in self.get_extract_info_keys():
                if k in inf:
                    if not isinstance(inf[k], torch.Tensor):
                        assign_val = torch.tensor(inf[k]).to(self.args.device)
                    else:
                        assign_val = inf[k]
                    self.add_data[k][self.step, i] = assign_val

        self.actions[self.step].copy_(ac_info.action)
        self.action_log_probs[self.step].copy_(ac_info.action_log_probs)
        self.value_preds[self.step].copy_(ac_info.value)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        for k in self.hidden_states:
            self.hidden_states[k][self.step + 1].copy_(ac_info.hxs[k])

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for k in self.ob_keys:
            if k is None:
                self.obs[0].copy_(self.obs[-1])
            else:
                self.obs[k][0].copy_(self.obs[k][-1])

        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        # bad_masks = [self.bad_masks[-1]]
        # for bm in self.bad_masks[1:]:
        #     bad_masks.append(bm)
        # self.bad_masks = torch.stack(bad_masks)

        for k in self.add_data:
            self.add_data[k][0].copy_(self.add_data[k][-1])

        for k in self.hidden_states:
            self.hidden_states[k][0].copy_(self.hidden_states[k][-1])

    def compute_returns(self, next_value):
        exp_rewards = self.rewards.repeat(1, 1, self.value_dim)
        gamma = self.args.gamma
        if self.args.use_proper_time_limits:
            # Use the "bad_masks" to properly account for early terminations.
            # This is the case in mujoco OpenAI gym tasks.
            if self.args.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        exp_rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = (
                        delta
                        + gamma * self.args.gae_lambda * self.masks[step + 1] * gae
                    )
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    # ((R_{t+1} * \gamma * M_{t+1}) + (R_t * B_{t+1}) +
                    # (1 - B_{t+1}) * V_t
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + exp_rewards[step]
                    ) * self.bad_masks[step + 1] + (
                        1 - self.bad_masks[step + 1]
                    ) * self.value_preds[
                        step
                    ]
        else:
            if self.args.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        exp_rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )

                    gae = (
                        delta
                        + gamma * self.args.gae_lambda * self.masks[step + 1] * gae
                    )
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + exp_rewards[step]
                    )

    def compute_advantages(self):
        advantages = self.returns[:-1] - self.value_preds[:-1]
        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages
        # return advantages.reshape(-1, self.value_dim)

    def get_generator(
        self, advantages=None, num_mini_batch=None, mini_batch_size=None, **kwargs
    ):
        if self.args.recurrent_policy:
            data_generator = self.recurrent_generator(advantages, num_mini_batch)
        else:
            data_generator = self.feed_forward_generator(
                advantages, num_mini_batch, mini_batch_size
            )
        return data_generator

    def get_rollout_data(self, advantages=None):
        gen = self.get_generator(advantages, num_mini_batch=1)
        return next(gen)

    def feed_forward_generator(
        self, advantages, num_mini_batch=None, mini_batch_size=None
    ):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    num_processes, num_steps, num_processes * num_steps, num_mini_batch
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )

        for indices in sampler:
            obs_batch = None
            other_obs_batch = {}
            for k, ob_shape in self.ob_keys.items():
                if k is None:
                    obs_batch = self.obs[:-1].view(-1, *ob_shape)[indices]
                elif k == self.args.policy_ob_key:
                    obs_batch = self.obs[k][:-1].view(-1, *ob_shape)[indices]
                else:
                    other_obs_batch[k] = self.obs[k][:-1].view(-1, *ob_shape)[indices]

            assert obs_batch is not None, f"Found not find {self.args.policy_ob_key}"

            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            rewards_batch = self.rewards.view(-1, 1)[indices]

            hidden_states_batch = {}
            for k in self.hidden_states:
                hidden_states_batch[k] = self.hidden_states[k][:-1].view(
                    -1, self.hidden_states[k].size(-1)
                )[indices]

            value_preds_batch = self.value_preds[:-1].view(-1, self.value_dim)[indices]
            return_batch = self.returns[:-1].view(-1, self.value_dim)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, self.value_dim)[
                indices
            ]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, self.value_dim)[indices]

            yield {
                "state": obs_batch,
                "other_state": other_obs_batch,
                "reward": rewards_batch,
                "hxs": hidden_states_batch,
                "action": actions_batch,
                "value": value_preds_batch,
                "return": return_batch,
                "mask": masks_batch,
                "prev_log_prob": old_action_log_probs_batch,
                "adv": adv_targ,
            }

    def get_np_tensors(self):
        """
        Helper method to get the "simple" data in the buffer as numpy arrays. The
        data ordering is preserved.
        """
        ob_shape = self.ob_keys[None]
        s = self.obs[:-1].view(-1, *ob_shape).numpy()
        n_s = self.obs[1:].view(-1, *ob_shape).numpy()
        mask = self.masks[1:].view(-1, 1).numpy()
        actions = self.actions.view(-1, self.actions.size(-1)).numpy()
        reward = self.rewards.view(-1, 1).numpy()
        return s, n_s, actions, reward, mask

    def get_scalars(self):
        s, n_s, a, r, m = self.get_np_tensors()
        return int(s[0]), int(n_s[0]), a[0, 0], r[0, 0], m[0, 0]

    def recurrent_generator(self, advantages, num_mini_batch):
        # Only called if args.recurrent_policy is True
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )

        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            hidden_states_batch = defaultdict(list)
            obs_batch = defaultdict(list)
            other_obs_batch = defaultdict(list)
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            reward_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                for k, ob_shape in self.ob_keys.items():
                    if k is None:
                        obs_batch[None].append(self.obs[:-1, ind])
                    elif k == self.args.policy_ob_key or k is None:
                        obs_batch[k].append(self.obs[k][:-1, ind])
                    else:
                        other_obs_batch[k].append(self.obs[k][:-1, ind])

                for k in self.hidden_states:
                    hidden_states_batch[k].append(self.hidden_states[k][0:1, ind])

                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                reward_batch.append(self.rewards[:, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            reward_batch = torch.stack(reward_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            for k in hidden_states_batch:
                hidden_states_batch[k] = torch.stack(hidden_states_batch[k], 1).view(
                    N, -1
                )
            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for k in obs_batch:
                obs_batch[k] = torch.stack(obs_batch[k], 1)
                obs_batch[k] = _flatten_helper(T, N, obs_batch[k])
            for k in other_obs_batch:
                other_obs_batch[k] = torch.stack(other_obs_batch[k], 1)
                other_obs_batch[k] = _flatten_helper(T, N, other_obs_batch[k])
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            reward_batch = _flatten_helper(T, N, reward_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            # No need to return obs dict if there's only one thing in
            # dictionary
            if len(obs_batch) == 1:
                obs_batch = next(iter(obs_batch.values()))

            yield {
                "other_state": other_obs_batch,
                "reward": reward_batch,
                "hxs": hidden_states_batch,
                "state": obs_batch,
                "action": actions_batch,
                "value": value_preds_batch,
                "return": return_batch,
                "mask": masks_batch,
                "prev_log_prob": old_action_log_probs_batch,
                "adv": adv_targ,
            }

    def get_actions(self):
        actions = self.actions.view(-1, self.actions.size(-1))
        return actions

    def get_obs(self, step):
        obs = {}
        for k in self.ob_keys:
            if k is None:
                return self.obs[step]
            obs[k] = self.obs[k][step]
        assert len(obs) != 0, "No matching keys in state observation dictionary"

        return obs

    def get_hidden_state(self, step):
        return rutils.deep_dict_select(self.hidden_states, step)

    def get_masks(self, step):
        return self.masks[step]
