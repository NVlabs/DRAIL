from pytorch_sac.replay_buffer import ReplayBuffer
from rlf.il.transition_dataset import TransitionDataset
from goal_prox.method.utils import trim_episodes_trans
import torch

class SqilReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, device, use_frac,
            batch_size, data_path):
        super().__init__(obs_shape, action_shape, capacity, device)

        expert_dataset = TransitionDataset(data_path, trim_episodes_trans)
        use_dataset = expert_dataset.compute_split(use_frac)
        self.expert_train_loader = torch.utils.data.DataLoader(
                dataset=use_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True)
        self.expert_batch_iter = None

    def get_next_expert_batch(self):
        batch = None
        if self.expert_batch_iter is not None:
            try:
                batch = next(self.expert_batch_iter)
            except StopIteration:
                pass

        if batch is None:
            self.expert_batch_iter = iter(self.expert_train_loader)
            batch = next(self.expert_batch_iter)
        return batch

    def sample(self, batch_size):
        obses, actions, rewards, next_obses, not_dones, not_dones_no_max = super().sample(batch_size)
        expert_sample = self.get_next_expert_batch()

        expert_masks = (1 - expert_sample['done']).to(self.device)
        expert_actions = expert_sample['actions'].to(self.device)
        expert_state = expert_sample['state'].to(self.device)
        expert_next_state = expert_sample['next_state'].to(self.device)

        not_dones = torch.cat([not_dones, expert_masks.unsqueeze(-1)], dim=0)
        not_dones_no_max = torch.cat([not_dones_no_max, expert_masks.unsqueeze(-1)], dim=0)
        actions = torch.cat([actions, expert_actions], dim=0)
        obses = torch.cat([obses, expert_state], dim=0)
        next_obses = torch.cat([next_obses, expert_next_state], dim=0)

        rewards = torch.cat([
            torch.zeros(rewards.shape).to(rewards.device),
            torch.ones(rewards.shape).to(rewards.device)
            ], dim=0)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
