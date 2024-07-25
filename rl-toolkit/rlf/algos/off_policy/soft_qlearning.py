from rlf.algos.off_policy.actor_critic_updater import ActorCriticUpdater
from rlf.algos.off_policy.off_policy_base import OffPolicy
import rlf.algos.utils as autils
import torch.nn.functional as F
import torch


def sample_actions(self, policy, state, add_info, n_particles):
    return torch.stack([policy.forward(state, *add_info)
        for _ in range(n_particles)])

class SoftQLearning(OffPolicy):
    """
    Implementation of https://arxiv.org/pdf/1702.08165.pdf
    """
    def init(self, policy, args):
        super().init(policy, args)
        self.target_policy = self._copy_policy()

    def update(self, storage):
        if len(storage) < self.args.batch_size:
            return {}

        state, n_state, action, reward, add_info, n_add_info = self._sample_transitions(storage)

        curQ = self.policy(state).gather(1, action.long())

        # Directly from Eq (5) for computing the value function
        nextQ = self.target_policy(n_state)
        nextV = self.args.alpha * torch.log(torch.sum(torch.exp(nextQ / self.args.alpha), dim=1))
        nextV = nextV.unsqueeze(1)
        target = reward + self.args.gamma * n_add_info['masks'] * nextV
        target = target.detach()

        loss = F.mse_loss(curQ, target)
        autils.soft_update(self.policy, self.target_policy, self.args.tau)

        return {
                'loss': loss.item()
                }


    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--tau',
            type=float,
            default=1e-3,
            help='Mixture for the target network weight update')
