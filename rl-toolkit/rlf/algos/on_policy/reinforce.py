from collections import defaultdict
import torch.optim as optim
from rlf.algos.on_policy.on_policy_base import OnPolicy


class REINFORCE(OnPolicy):
    def update(self, rollouts):
        sample = rollouts.get_rollout_data()
        ac_eval = self.policy.evaluate_actions(sample['state'],
                sample['hxs'], sample['mask'],
                sample['action'])

        loss = -ac_eval['log_prob'] * sample['return']
        loss = loss.sum()

        self._standard_step(loss)
        return {
                'loss': loss.item()
                }
