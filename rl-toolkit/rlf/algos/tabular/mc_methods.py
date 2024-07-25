from rlf.algos.tabular.base_tabular import BaseTabular
from rlf.policies.base_policy import get_step_info
import numpy as np
from collections import defaultdict

class TabularMcMethods(BaseTabular):

    def init(self, policy, args):
        super().init(policy, args)
        self.returns = defaultdict(lambda: [[]
            for a in range(policy.action_space.n)])

    def on_traj_finished(self, traj):
        # G is computed backwards
        G = 0
        for i in reversed(range(len(traj))):
            s, a, mask, info, r = traj[i][0]
            G = self.args.gamma * G + r
            self.returns[s][a].append(G)
            self.policy.Q[s][a] = np.mean(self.returns[s][a])

    def update(self, rollout):
        super().update(rollout)
        return {}

    def get_add_args(self, parser):
        super().get_add_args(parser)
