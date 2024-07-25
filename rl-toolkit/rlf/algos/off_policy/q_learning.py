from rlf.algos.off_policy.off_policy_base import OffPolicy
import rlf.algos.utils as autils
import torch
import torch.nn.functional as F
import torch.nn as nn


class QLearning(OffPolicy):
    """
    Q(s_t, a_t) target:
    r_t + \gamma * max_a Q(s_{t+1}, a)
    """

    def init(self, policy, args):
        super().init(policy, args)
        self.target_policy = self._copy_policy()

    def update(self, storage):
        if len(storage) < self.args.batch_size:
            return {}

        for update_i in range(self.args.updates_per_batch):
            state, n_state, action, reward, _, n_add_info = self._sample_transitions(storage)

            next_q_vals = self.target_policy(n_state).max(1)[0].detach().unsqueeze(-1) * n_add_info['masks']
            target = reward + (next_q_vals * self.args.gamma)

            cur_q_vals = self.policy(state).gather(1, action)
            loss = F.mse_loss(cur_q_vals.view(-1), target.view(-1))

            self._standard_step(loss)

        autils.soft_update(self.policy, self.target_policy, self.args.tau)

        return {
                'loss': loss.item()
                }


    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--tau', type=float, default=1.0,
            help=("Mixture for the target network weight update. ",
                "If one this is regular DQN (no target network)"))

        parser.add_argument('--updates-per-batch', type=int, default=1,
            help='Number of updates to perform in each call to update')
