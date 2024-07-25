from rlf.algos.tabular.policy_iteration import PolicyIteration
import numpy as np


class ValueIteration(PolicyIteration):
    def policy_eval(self):
        gam = self.args.gamma
        for s in self.states:
            q_vals = np.zeros((len(self.actions)))
            for a in self.actions:
                probs = self.P[s][self.policy.pi[s]]
                for prob, n_state, reward, done in probs:
                    q_vals[a] += prob * (reward + gam * self.value[n_state])
            self.value[s] = np.max(q_vals)
