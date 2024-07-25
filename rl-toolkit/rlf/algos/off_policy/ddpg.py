from rlf.algos.off_policy.actor_critic_updater import ActorCriticUpdater
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import rlf.algos.utils as autils
from collections import defaultdict


class DDPG(ActorCriticUpdater):
    def init(self, policy, args):
        if args.updates_per_batch is None:
            args.updates_per_batch = args.update_every
        super().init(policy, args)

    def update(self, storage):
        super().update(storage)
        if len(storage) < self.args.warmup_steps:
            return {}

        if len(storage) < self.args.batch_size:
            return {}

        ns = self.get_completed_update_steps(self.update_i)
        if ns % self.args.update_every != 0:
            return {}

        avg_log_vals = defaultdict(list)
        for i in range(self.args.updates_per_batch):
            log_vals = self._optimize(*self._sample_transitions(storage))
            for k,v in log_vals.items():
                avg_log_vals[k].append(v)

        avg_log_vals = {k: np.mean(v) for k,v in avg_log_vals.items()}

        return avg_log_vals


    def _optimize(self, state, n_state, action, reward, add_info, n_add_info):
        n_masks = n_add_info['masks']
        n_masks = n_masks.to(self.args.device)

        # Get the Q-target
        n_action = self.target_policy(n_state, **n_add_info)
        next_q = self.target_policy.get_value(n_state, n_action, **n_add_info)
        next_q *= n_masks
        target = (reward + (self.args.gamma * next_q)).detach()

        # Compute the critic loss. (Just a TD loss)
        q = self.policy.get_value(state, action, **add_info)
        critic_loss = F.mse_loss(q.view(-1), target.view(-1))
        self._standard_step(critic_loss, 'critic_opt')

        # Compute the actor loss
        choose_action = self.policy(state, **add_info)
        actor_loss = -self.policy.get_value(state, choose_action, **add_info).mean()
        self._standard_step(actor_loss, 'actor_opt')

        if self.update_i % self.args.target_delay == 0:
            autils.soft_update(self.policy, self.target_policy, self.args.tau)

        return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item()
                }


    def get_add_args(self, parser):
        super().get_add_args(parser)

        #########################################
        # Overrides
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--batch-size', type=int, default=int(64))

        #########################################
        # New args
        parser.add_argument('--warmup-steps',
            type=int,
            default=int(1000),
            help='Number of steps before any updates are applied')

        parser.add_argument('--target-delay',
            type=int,
            default=1, help="""
            Frequency of updating the target network
            """)

        parser.add_argument('--update-every',
            type=int,
            default=int(50),
            help="""
                Update frequency. If 50 this will only actually the network
                every 50 calls to the update function
                """)

        parser.add_argument('--updates-per-batch',
            type=int,
            default=None,
            help="""
            The number of epochs per update. Defaults to `update-every`
            """)
