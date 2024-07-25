from rlf.algos.tabular.base_tabular import BaseTabular
import numpy as np


class PolicyIteration(BaseTabular):
    def init(self, policy, args):
        super().init(policy, args)

    def get_num_updates(self):
        return self.num_updates

    def ac_to_onehot(self, x):
        one_hot = np.zeros((len(x), len(self.actions)))
        one_hot[np.arange(len(x)), x] = 1
        return one_hot

    def update(self, storage):
        prev_pi = np.array(self.policy.pi)
        for iter_i in range(self.args.iters_per_update):
            self.policy_eval()
            self.policy_improvement()

        delta = np.max(np.abs(
            self.ac_to_onehot(self.policy.pi) - \
            self.ac_to_onehot(prev_pi)))
        return {
                '_pr_delta': delta
                }

    def set_env_ref(self, envs):
        super().set_env_ref(envs)
        self.P = envs.venv.envs[0].env.env.env.P
        self.states = list(range(self.policy.obs_space.n))
        self.actions = list(range(self.policy.action_space.n))
        self.value = np.zeros((len(self.states)))

    def policy_eval(self):
        gam = self.args.gamma
        for s in self.states:
            probs = self.P[s][self.policy.pi[s]]
            new_value = 0
            for prob, n_state, reward, done in probs:
                new_value += prob * (reward + gam * self.value[n_state])
            self.value[s] = new_value

    def policy_improvement(self):
        gam = self.args.gamma
        for s in self.states:
            q_vals = np.zeros((len(self.actions)))
            for a in self.actions:
                for prob, n_state, reward, done in self.P[s][a]:
                    q_vals[a] += prob * (reward + gam * self.value[n_state])

            self.policy.pi[s] = np.argmax(q_vals)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        # No environment interaction
        parser.add_argument('--num-steps', type=int, default=0)
        parser.add_argument('--num-iters', type=int, default=100)
        parser.add_argument('--iters-per-update', type=int, default=1)
