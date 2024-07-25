# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import rlf.algos.utils as autils
import torch.nn as nn
import torch.optim as optim
from rlf.algos.base_algo import BaseAlgo
from rlf.args import str2bool


class BaseNetAlgo(BaseAlgo):
    def __init__(self):
        super().__init__()
        self.arg_prefix = ""

    def _arg(self, k):
        prefix_str = self.arg_prefix.replace("-", "_")
        return self.arg_vars[prefix_str + k]

    def init(self, policy, args):
        super().init(policy, args)
        self.arg_vars = vars(args)
        self._optimizers = self._get_optimizers()
        self.obs_space = policy.obs_space
        self.action_space = policy.action_space

        if self._arg("linear_lr_decay"):
            if self._arg("lr_env_steps") is None:
                self.lr_updates = self.get_num_updates()
            else:
                self.lr_updates = (
                    int(self._arg("lr_env_steps"))
                    // args.num_steps
                    // args.num_processes
                )

    def get_optimizer(self, opt_key: str):
        return self._optimizers[opt_key][0]

    def update(self, storage):
        log_vals = super().update(storage)
        for k, (opt, _, initial_lr) in self._optimizers.items():
            lr = None
            for param_group in opt.param_groups:
                lr = param_group["lr"]
            log_vals[k + "_lr"] = lr
        return log_vals

    def _copy_policy(self):
        cp_policy = super()._copy_policy()
        if next(self.policy.parameters()).is_cuda:
            cp_policy = cp_policy.cuda()
        autils.hard_update(self.policy, cp_policy)
        return cp_policy

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        # Load the optimizers where they left off.
        for k, (opt, _, _) in self._optimizers.items():
            opt.load_state_dict(checkpointer.get_key(k))

    def save(self, checkpointer):
        super().save(checkpointer)
        for k, (opt, _, _) in self._optimizers.items():
            checkpointer.save_key(k, opt.state_dict())

    def load(self, checkpointer):
        super().load(checkpointer)
        for k, (opt, _, _) in self._optimizers.items():
            if checkpointer.has_load_key(k):
                opt.load_state_dict(checkpointer.get_key(k))

    def pre_update(self, cur_update):
        super().pre_update(cur_update)
        if self._arg("linear_lr_decay"):
            for k, (opt, _, initial_lr) in self._optimizers.items():
                autils.linear_lr_schedule(cur_update, self.lr_updates, initial_lr, opt)

    def _clip_grad(self, params):
        """
        Helper function to clip gradients
        """
        if self._arg("max_grad_norm") > 0:
            nn.utils.clip_grad_norm_(params, self._arg("max_grad_norm"))

    def _standard_step(self, loss, optimizer_key="actor_opt"):
        """
        Helper function to compute gradients, clip gradients and then take
        optimization step.
        """
        opt, get_params_fn, _ = self._optimizers[optimizer_key]
        opt.zero_grad()
        loss.backward()
        self._clip_grad(get_params_fn())
        opt.step()

    def set_arg_prefix(self, arg_prefix):
        self.arg_prefix = arg_prefix + "-"

    def get_add_args(self, parser):
        """
        Adds default arguments that might be useful for all algorithms that
        update neural networks. Added arguments:
        * --max-grad-norm
        * --linear-lr-decay
        * --eps
        * --lr
        All can be prefixed with `self.arg_prefix`.
        """
        super().get_add_args(parser)
        parser.add_argument(
            f"--{self.arg_prefix}max-grad-norm",
            default=0.5,
            type=float,
            help="-1 results in no grad norm",
        )
        parser.add_argument(
            f"--{self.arg_prefix}linear-lr-decay", type=str2bool, default=True
        )
        parser.add_argument(
            f"--{self.arg_prefix}lr-env-steps",
            type=float,
            default=None,
            help="only used for lr schedule",
        )
        parser.add_argument(
            f"--{self.arg_prefix}eps",
            type=float,
            default=1e-12,
            help="""
                            optimizer epsilon (default: 1e-5)
                            NOTE: The PyTorch default is 1e-8 see
                            https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
                            """,
        )
        parser.add_argument(
            f"--{self.arg_prefix}lr",
            type=float,
            default=1e-3,
            help="learning rate (default: 1e-3)",
        )

    @staticmethod
    def _create_opt(module_to_opt: nn.Module, lr: float, eps: float = 1e-8):
        get_params_fn = lambda: module_to_opt.parameters()
        return (optim.Adam(get_params_fn(), lr=lr, eps=eps), get_params_fn, lr)

    def _get_optimizers(self):
        return {
            "actor_opt": BaseNetAlgo._create_opt(
                self.policy, self._arg("lr"), self._arg("eps")
            )
        }
