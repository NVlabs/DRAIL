from rlf.algos.tabular.base_tabular import BaseTabular
from rlf.policies.base_policy import get_step_info
import numpy as np

class TabularTdMethods(BaseTabular):
    def update(self, rollout):
        super().update(rollout)
        s, n_s, a, r, mask = rollout.get_scalars()

        step_info = get_step_info(self.update_i, 0, None, self.args)

        if self.args.td_alg == 'sarsa':
            n_a = self.policy.get_action(n_s, step_info=step_info).take_action
            next_q = self.policy.Q[n_s][n_a]
        elif self.args.td_alg == 'exp-sarsa':
            action_probs = self.policy.get_action_probs(n_s, step_info)
            next_q = np.dot(action_probs, self.policy.Q[n_s])
        elif self.args.td_alg == 'qlearn':
            next_q = np.max(self.policy.Q[n_s])
        else:
            raise ValueError('Unregonized algo %s' % self.args.td_alg)

        target = r + self.args.gamma * next_q
        target *= mask
        tderr = target - self.policy.Q[s][a]
        self.policy.Q[s][a] += self.args.lr * tderr
        return {
                '_pr_td_error': tderr
                }

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--td-alg', type=str, default='sarsa')
