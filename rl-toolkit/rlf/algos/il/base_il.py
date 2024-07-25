# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import os.path as osp

import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.algos.base_net_algo import BaseNetAlgo
from rlf.il.transition_dataset import TransitionDataset
from rlf.rl import utils


class ExperienceGenerator(object):
    def init(self, policy, args, exp_gen_num_trans):
        pass

    def get_batch(self):
        pass

    def reset(self):
        pass


class BaseILAlgo(BaseNetAlgo):
    def __init__(self, exp_generator=None):
        super().__init__()
        self.exp_generator = exp_generator
        self.data_iter = None
        self._holdout_idxs = None
        # By default do not transform the dataset at all.
        self._transform_dem_dataset_fn = None

    def set_transform_dem_dataset_fn(self, transform_dem_dataset_fn):
        self._transform_dem_dataset_fn = transform_dem_dataset_fn

    def _load_expert_data(self, policy, args):
        assert args.traj_load_path is not None, "Must specify expert demonstrations!"
        self.args = args
        self.orig_dataset = self._get_traj_dataset(
            osp.join(args.cwd, args.traj_load_path)
        )
        self.orig_dataset = self.orig_dataset.to(args.device)
        num_trajs = self._create_train_loader(args)

        trans_count_str = utils.human_format_int(
            len(self.expert_train_loader) * args.traj_batch_size
        )
        print("Loaded %s transitions for imitation" % trans_count_str)
        print("(%i trajectories)" % num_trajs)

    @property
    @functools.lru_cache()
    def expert_stats(self):
        return self.orig_dataset.get_expert_stats(self.args.device)

    def _create_train_loader(self, args):
        # Always keep track of the non-shuffled, non-split version of the
        # dataset.
        self.expert_dataset = self.orig_dataset
        if args.traj_frac != 1.0:
            self.expert_dataset = self.expert_dataset.compute_split(
                args.traj_frac, args.seed
            )
        if args.traj_viz:
            self.expert_dataset.viz(args)

        if args.traj_val_ratio != 0.0:
            N = len(self.expert_dataset)
            n_val = int(N * args.traj_val_ratio)

            train_dataset, val_dataset = torch.utils.data.random_split(
                self.expert_dataset,
                [N - n_val, n_val],
                torch.Generator().manual_seed(self.args.seed),
            )
            val_traj_batch_size = min(len(val_dataset), self.args.traj_batch_size)
            self.val_train_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=val_traj_batch_size,
                shuffle=True,
                drop_last=True,
            )
        else:
            train_dataset = self.expert_dataset
            self.val_train_loader = None

        self.expert_train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.traj_batch_size,
            shuffle=True,
            drop_last=True,
        )

        if isinstance(self.expert_dataset, torch.utils.data.Subset):
            return int(args.traj_frac * self.expert_dataset.dataset.get_num_trajs())
        else:
            return self.expert_dataset.get_num_trajs()

    def _get_next_data(self):
        if self.exp_generator is None:
            if self.data_iter is None:
                self.data_iter = iter(self.expert_train_loader)                
            try:
                return next(self.data_iter, None)
            except IndexError:
                return None
        else:
            import ipdb; ipdb.set_trace()
            return self.exp_generator.get_batch()

    def _reset_data_fetcher(self):
        if self.exp_generator is None:
            self.data_iter = iter(self.expert_train_loader)
        else:
            self.exp_generator.reset()

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.il_out_action_norm:
            print("Setting environment action denormalization")
            settings.action_fn = self._denorm_action
        return settings

    def _adjust_action(self, x):
        if not self.args.il_in_action_norm:
            return x
        return (x) / (self.expert_stats["action"][1] + 1e-8)

    def _denorm_action(self, x):
        return (x) * self.expert_stats["action"][1]

    def init(self, policy, args):
        # Load the expert data first, so we can calculate the needed number of
        # steps for the learning rate scheduling in base_net_algo.py.
        if self.exp_generator is None:
            self._load_expert_data(policy, args)
        else:
            self.exp_generator.init(policy, args, args.exp_gen_num_trans)
            print(f"Generating {args.exp_gen_num_trans} transitions for imitation")
        super().init(policy, args)

    def _get_expert_traj_stats(self):
        return self.expert_mean, self.expert_std

    def _get_traj_dataset(self, traj_load_path):
        return TransitionDataset(traj_load_path, self._transform_dem_dataset_fn)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--traj-load-path", type=str, default=None)
        parser.add_argument("--traj-batch-size", type=int, default=128)
        parser.add_argument(
            "--traj-val-ratio",
            type=float,
            default=0.0,
            help="""
                Ratio of the dataset which is used for validation. This is only
                supported for algorithms where there is some supervised
                objective and this makes sense (i.e. for
                something like BC where training is performed offline).
                """,
        )
        parser.add_argument(
            "--traj-frac",
            type=float,
            default=1.0,
            help="The fraction of trajectories to use",
        )
        parser.add_argument("--traj-viz", action="store_true", default=False)
        parser.add_argument("--exp-gen-num-trans", type=int, default=None)

        # Unless you have some weird dataset situation, you probably want to
        # specify either both or none of these. Specifying only in-action-norm
        # will normalize the actions as input to the policy but will not
        # denormalize the output when being passed to the environment.
        parser.add_argument(
            "--il-in-action-norm",
            action="store_true",
            default=False,
            help="Normalize expert actions input to the policy",
        )
        parser.add_argument(
            "--il-out-action-norm",
            action="store_true",
            default=False,
            help="Denormalize actions in the environment",
        )
