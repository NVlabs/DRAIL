from rlf.algos.tabular.base_tabular import BaseTabular
from rlf.policies.tabular.action_value_policy import ActionValuePolicy
import numpy as np

class SimpleBanditAlgo(BaseTabular):
    def init(self, policy, args):
        assert isinstance(policy, ActionValuePolicy)
        super().init(policy, args)
        self.counts = np.zeros(policy.Q.shape)

    def update(self, storage):
        Q = self.policy.Q
        a = storage.actions.item()
        R = storage.rewards.item()

        if self.args.lr is None:
            use_lr = (1.0 / self.counts[a])
        else:
            use_lr = self.args.lr
        self.counts[a] += 1

        self.policy.Q[a] = Q[a] + use_lr * (R - Q[a])

        return {}

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--lr', type=float, default=None)

