from rlf.algos.il.gail import GailDiscrim
import torch
import torch.nn as nn
from functools import partial
import rlf.il.utils as iutils
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.il.transition_dataset import TransitionDataset
import rlf.rl.utils as rutils
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from rlf.rl.model import ConcatLayer
from collections import defaultdict
import torch.nn.functional as F

class GAIFO(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([GaifoDiscrim(get_discrim), agent_updater], 1)

class PairTransitionDataset(TransitionDataset):
    def __init__(self, load_path, transform_dem_dataset_fn):
        super().__init__(load_path, transform_dem_dataset_fn)
        self.trajs['next_obs'] = self.trajs['next_obs'].float()

    def __getitem__(self, i):
        return {
                'state': self.trajs['obs'][i],
                'next_state': self.trajs['next_obs'][i],
                'done': self.trajs['done'][i],
                }

def get_default_discrim(ac_dim, in_shape):
    """
    Returns: (nn.Module) Should take two states as input.
    """
    hidden_dim = 64
    return nn.Sequential(
        ConcatLayer(-1),
        nn.Linear(2*in_shape[0], hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, 1))

class DoubleStateDiscrim(nn.Module):
    def __init__(self, state_enc, hidden_dim=64):
        super().__init__()
        self.state_enc = state_enc
        output_size = self.state_enc.output_shape[0]
        self.head = nn.Sequential(
                nn.Linear(output_size, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1))

    def forward(self, s0, s1):
        both_s = torch.cat([s0, s1], dim=1)
        both_s_enc, _ = self.state_enc(both_s, None, None)
        return self.head(both_s_enc)


class GaifoDiscrim(GailDiscrim):
    def _create_discrim(self):
        new_shape = list(rutils.get_obs_shape(self.policy.obs_space))
        new_shape[0] *= 2
        base_net = self.policy.get_base_net_fn(new_shape)
        return DoubleStateDiscrim(base_net).to(self.args.device)

    def _get_traj_dataset(self, traj_load_path):
        return PairTransitionDataset(traj_load_path, self._transform_dem_dataset_fn)

    def _trans_batches(self, expert_batch, agent_batch):
        agent_batch = iutils.select_idx_from_dict(agent_batch,
                self.agent_obs_pairs)
        return expert_batch, agent_batch

    def _compute_discrim_loss(self, agent_batch, expert_batch, obsfilt):
        d = self.args.device
        exp_s0 = self._norm_expert_state(expert_batch['state'],
                obsfilt).float()
        exp_s1 = self._norm_expert_state(expert_batch['next_state'],
                obsfilt).float()

        agent_s0 = agent_batch['state'].to(d)
        agent_s1 = agent_batch['next_state'].to(d)

        expert_d = self.discrim_net(exp_s0, exp_s1)
        agent_d = self.discrim_net(agent_s0, agent_s1)
        # Select the valid transitions where the episode did not end in
        # between.
        expert_d = expert_d[expert_batch['done'] == 0]
        agent_d = agent_d[agent_batch['mask'] == 1]
        return expert_d, agent_d, 0

    def _compute_disc_val(self, state, next_state, action):
        return self.discrim_net(state, next_state)

    def _get_sampler(self, storage):
        obs = storage.get_def_obs_seq()
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        self.agent_obs_pairs = {
                'state': obs[:-1].view(-1, *ob_shape),
                'next_state': obs[1:].view(-1, *ob_shape),
                'mask': storage.masks[:-1].view(-1, 1),
                }
        failure_sampler = BatchSampler(SubsetRandomSampler(
            range(self.args.num_steps)), self.args.traj_batch_size,
            drop_last=True)
        return self.expert_train_loader, failure_sampler

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        settings.include_info_keys.extend([
            ('final_obs', lambda env: rutils.get_obs_shape(env.observation_space))
            ])
        return settings

    def _compute_discrim_reward(self, storage, step, add_info):
        state = rutils.get_def_obs(storage.get_obs(step))

        next_state = rutils.get_def_obs(storage.get_obs(step+1))
        masks = storage.masks[step+1]
        finished_episodes = [i for i in range(len(masks)) if masks[i] == 0.0]
        add_inputs = {k: v[(step+1)-1] for k,v in add_info.items()}
        obsfilt = self.get_env_ob_filt()
        for i in finished_episodes:
            next_state[i] = add_inputs['final_obs'][i]
            if obsfilt is not None:
                next_state[i] = torch.FloatTensor(obsfilt(next_state[i].cpu().numpy(),
                        update=False)).to(self.args.device)

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

    def _update_reward_func(self, storage):
        self.discrim_net.train()

        d = self.args.device
        log_vals = defaultdict(lambda: 0)
        obsfilt = self.get_env_ob_filt()

        n = 0
        expert_sampler, agent_sampler = self._get_sampler(storage)
        for epoch_i in range(self.args.n_gail_epochs):
            for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
                expert_batch, agent_batch = self._trans_batches(
                    expert_batch, agent_batch)
                n += 1
                expert_d, agent_d, grad_pen = self._compute_discrim_loss(agent_batch, expert_batch,
                        obsfilt)
                if self.args.reward_type == 'gaifo':
                    expert_loss = F.binary_cross_entropy_with_logits(expert_d,
                                                                     torch.zeros(expert_d.shape).to(d))
                    agent_loss = F.binary_cross_entropy_with_logits(agent_d,
                                                                    torch.ones(agent_d.shape).to(d))
                else:
                    expert_loss = F.binary_cross_entropy_with_logits(expert_d,
                                                                     torch.ones(expert_d.shape).to(d))
                    agent_loss = F.binary_cross_entropy_with_logits(agent_d,
                                                                    torch.zeros(agent_d.shape).to(d))
                discrim_loss = expert_loss + agent_loss

                self.opt.zero_grad()
                discrim_loss.backward()
                self.opt.step()

                log_vals['discrim_loss'] += discrim_loss.item()
                log_vals['expert_loss'] += expert_loss.item()
                log_vals['agent_loss'] += agent_loss.item()

        for k in log_vals:
            log_vals[k] /= n

        return log_vals
