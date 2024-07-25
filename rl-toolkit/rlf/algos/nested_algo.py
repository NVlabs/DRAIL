from typing import Optional

from rlf.algos.base_algo import BaseAlgo


class NestedAlgo(BaseAlgo):
    def __init__(
        self,
        modules,
        designated_rl_idx,
        designated_settings_idx=0,
    ):
        """
        :param modules: ([rlf.algos.base_algo.BaseAlgo])
        :param designated_rl_idx: (int) which module to use to return
          storage object and other things for the RL training loop.
        :param designated_settings_idx: (int) which module to use to return
          algorithm settings which are used in environment creation amoung
          other things.
        """
        self.modules = modules
        self.designated_rl_idx = designated_rl_idx
        self.designated_settings_idx = designated_settings_idx

    def init(self, policy, args):
        for module in self.modules:
            module.init(policy, args)

    def get_steps_generator(self, update_iter):
        return self.modules[self.designated_rl_idx].get_steps_generator(update_iter)

    def get_num_updates(self):
        return self.modules[self.designated_rl_idx].get_num_updates()

    def set_env_ref(self, envs):
        for module in self.modules:
            module.set_env_ref(envs)

    def get_completed_update_steps(self, num_updates):
        n_updates = self.modules[0].get_completed_update_steps(num_updates)
        for m in self.modules[1:]:
            if m.get_completed_update_steps(num_updates) != n_updates:
                raise ValueError(
                    "All submodules must return the same number of updates"
                )
        return n_updates

    def set_get_policy(self, get_policy_fn, policy_args):
        for module in self.modules:
            module.set_get_policy(get_policy_fn, policy_args)

    def _copy_policy(self):
        raise NotImplementedError("Need to implement copy policy for NestedAlgo")

    def load_resume(self, checkpointer):
        for module in self.modules:
            module.load_resume(checkpointer)

    def load(self, checkpointer):
        for module in self.modules:
            module.load(checkpointer)

    def save(self, checkpointer):
        for module in self.modules:
            module.save(checkpointer)

    def pre_update(self, cur_update):
        for module in self.modules:
            module.pre_update(cur_update)

    def get_storage_buffer(self, policy, envs, args):
        return self.modules[self.designated_rl_idx].get_storage_buffer(
            policy, envs, args
        )

    def get_env_settings(self, args):
        return self.modules[self.designated_settings_idx].get_env_settings(args)

    def get_add_args(self, parser):
        for module in self.modules:
            module.get_add_args(parser)

    def on_traj_finished(self, traj):
        for module in self.modules:
            module.on_traj_finished(traj)

    def first_train(self, log, eval_policy, env_interface):
        for module in self.modules:
            module.first_train(log, eval_policy, env_interface)

    def update(self, storage, args=None, beginning=False, t=1, adm_first_freeze=False):
        log_vals = {}
        if adm_first_freeze and t <= 0.1:
            module = self.modules[1]
            add_log_vals = module.update(storage, beginning, t)
            log_vals = {**log_vals, **add_log_vals}
        else:
            # modules : [adm.adm.ADMDiscrim] and [rlf.algos.on_policy.ppo.PPO]
            if args.freeze_policy:
                add_log_vals = self.modules[0].update(storage, beginning, t)
                log_vals = {**log_vals, **add_log_vals}
            else:
                for module in self.modules:
                    add_log_vals = module.update(storage, beginning, t)
                    log_vals = {**log_vals, **add_log_vals}
        return log_vals
