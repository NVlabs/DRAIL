from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.il.bc import BehavioralCloning
import rlf.rl.utils as rutils
import copy
import torch

class BehavioralCloningPretrain(NestedAlgo):
    def __init__(self, modules, designated_rl_idx, designated_settings_idx=0):
        self.bc = BehavioralCloning(set_arg_defs=False)
        super().__init__(modules, designated_rl_idx, designated_settings_idx)

    def init(self, policy, args):
        self.args = args
        self.policy = policy
        super().init(policy, args)
        self.bc.init(policy, args)

    def first_train(self, log, eval_policy, env_interface):
        rutils.pstart_sep()
        print('Pre-training policy with BC')
        self.bc.full_train()

        bc_eval_args = copy.copy(self.args)
        bc_eval_args.eval_num_processes = 32
        bc_eval_args.num_eval = 5
        bc_eval_args.num_render = 0
        tmp_env = eval_policy(self.policy, 0, bc_eval_args)
        if tmp_env is not None:
            tmp_env.close()

        rutils.pend_sep()

        super().first_train(log, eval_policy, env_interface)

    def get_add_args(self, parser):
        self.bc.get_add_args(parser)
        super().get_add_args(parser)

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        self.bc_env_settings = self.bc.get_env_settings(args)
        #settings.state_fn = self.bc_env_settings.state_fn
        return settings

    def set_env_ref(self, envs):
        super().set_env_ref(envs)

        raise ValueError("Env ref code is depricated, please see for how to fix")
        #if env_norm is not None and self.bc_env_settings.state_fn is not None \
        #        and env_norm.ob_rms_dict is not None:
        #    # Set the normalization of the environment for further RL training.
        #    #state_stats = self.bc.expert_stats['state']
        #    if None in env_norm.ob_rms_dict:
        #        ob_rms = env_norm.ob_rms_dict[None]
        #    else:
        #        ob_rms = env_norm.ob_rms_dict['observation']
        #    ob_rms.mean = self.bc.norm_mean.cpu().float().numpy()
        #    # Conversion from std to var
        #    ob_rms.var = self.bc.norm_var.cpu().float().numpy()
        #    ob_rms.count += len(self.bc.expert_dataset)

    def update(self, storage):
        return super().update(storage)


