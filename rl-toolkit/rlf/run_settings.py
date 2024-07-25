import argparse
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from gym.spaces import Box

import rlf
import rlf.rl.utils as rutils
from rlf.args import get_default_parser
from rlf.envs.env_interface import get_env_interface
from rlf.exp_mgr import config_mgr
from rlf.il.traj_mgr import TrajSaver
from rlf.rl.checkpointer import Checkpointer
from rlf.rl.envs import make_vec_envs
from rlf.rl.evaluation import full_eval
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.rl.runner import Runner

def init_seeds(args):
    # Set all seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.set_num_threads(1)


try:
    from ray import tune

    MasterClass = tune.Trainable
except:

    class BlankTrainable:
        def __init__(self, config, logger_creator):
            pass

    MasterClass = BlankTrainable


class RunSettings(MasterClass):
    """
    Sets up the training, environments, and all other information needed for
    running an algorithm.
    """

    def __init__(self, args_str: str = None, config=None, logger_creator=None):
        """
        Args:
        :param args_str: Parse the arguments from this string.
        :param config: The config for tune.Trainable if used.
        :param logger_creator: Also used for tune.Trainble if used.
        """
        self._preset_args = None if args_str is None else args_str.split(" ")
        self.working_dir = os.getcwd()

        base_parser = self._get_base_parser()
        if self._preset_args is None:
            self.base_args, _ = base_parser.parse_known_args()
        else:
            self.base_args, _ = base_parser.parse_known_args(self._preset_args)
        super().__init__(config, logger_creator)

    def _get_base_parser(self):
        base_parser = argparse.ArgumentParser()
        self.get_add_args(base_parser)
        return base_parser

    def get_config_file(self) -> str:
        """
        :return: The location to a config file that holds whatever information
            about the project.
        """
        return osp.join(self.working_dir, "config.yaml")

    def create_traj_saver(self, save_path: str) -> rlf.il.TrajSaver:
        """
        How trajectories should be saved if desired.
        :save_path: file name to write the trajectories to
        """
        return TrajSaver(save_path)

    def get_add_args(self, parser):
        pass

    def get_logger(self):
        return BaseLogger()

    def get_add_ray_config(self, config):
        return config

    def get_add_ray_kwargs(self):
        return {}

    def get_policy(self) -> rlf.policies.BasePolicy:
        """
        :return: The policy for training
        """
        raise NotImplementedError("Must return policy to be used.")

    def get_algo(self) -> rlf.algos.BaseAlgo:
        """
        :return: The algorithm to update the policy with.
        """
        raise NotImplementedError("Must return algorithm to be used")

    def _get_env_interface(self, args, task_id=None):
        env_interface = get_env_interface(args.env_name)(args)
        env_interface.setup(args, task_id)
        return env_interface

    def get_parser(self):
        return get_default_parser()

    def get_args(self, algo, policy):
        parser = self.get_parser()
        algo.get_add_args(parser)
        policy.get_add_args(parser)

        if self._preset_args is None:
            args, rest = parser.parse_known_args()
        else:
            args, rest = parser.parse_known_args(self._preset_args)

        env_parser = argparse.ArgumentParser()
        get_env_interface(args.env_name)(args).get_add_args(env_parser)
        env_args, rest = env_parser.parse_known_args(rest)
        # Assign the env args to the main args namespace.
        rutils.update_args(args, vars(env_args))

        # Check that there are no arguments not accounted for in `base_args`
        _, rest_of_args = self._get_base_parser().parse_known_args(rest)
        if "-v" in rest_of_args:
            del rest_of_args[rest_of_args.index("-v")]
            print("Env args:")
            env_parser.print_help()
            print("Alg args:")
            parser.print_help()
            sys.exit(0)
        if len(rest_of_args) != 0:
            raise ValueError("Unrecognized arguments %s" % str(rest_of_args))

        # Convert the types of some of the standard types that don't allow the
        # scientific notation when expecting integer inputs.
        args.num_env_steps = int(args.num_env_steps)
        return args

    def stop(self):
        self.ray_runner.close()
        del self.ray_runner
        del self.ray_args

    def _sys_setup(self, add_args, ray_create, algo, policy):
        # Set up args used for training
        args = self.get_args(algo, policy)
        args.cwd = self.working_dir
        if "wandb" in add_args:
            del add_args["wandb"]
        rutils.update_args(args, add_args, True)
        if "cwd" in add_args:
            self.working_dir = add_args["cwd"]

        config_mgr.init(self.get_config_file())
        if args.ray:
            # No logger when ray is tuning
            log = BaseLogger()
        else:
            if ray_create:
                return None, None
            log = self.get_logger()
        for k, v in vars(self.base_args).items():
            if k not in args:
                setattr(args, k, v)
        log.init(args)
        log.set_prefix(args)

        args.device = torch.device("cuda:0" if args.cuda else "cpu")
        init_seeds(args)
        if args.detect_nan:
            torch.autograd.set_detect_anomaly(True)
        return args, log

    def create_runner(self, add_args={}, ray_create=False) -> rlf.Runner:
        """
        Gets the runner used for training.
        """
        policy = self.get_policy()
        algo = self.get_algo()

        args, log = self._sys_setup(add_args, ray_create, algo, policy)
        if args is None:
            return None
        env_interface = self._get_env_interface(args)

        checkpointer = Checkpointer(args)

        alg_env_settings = algo.get_env_settings(args)
        print("args.eval_only:", args.eval_only)
        #import ipdb; ipdb.set_trace()
        # Setup environment
        _, envs = make_vec_envs(
            args.env_name,
            args.seed,
            args.num_processes,
            args.gamma,
            args.device,
            True,
            env_interface,
            args,
            alg_env_settings,
            #set_eval=args.eval_only,
            False
        )

        rutils.pstart_sep()
        print("Action space:", envs.action_space)
        if isinstance(envs.action_space, Box):
            print("Action range:", (envs.action_space.low, envs.action_space.high))
        print("Observation space", envs.observation_space)
        rutils.pend_sep()

        # Setup policy
        policy_args = (envs.observation_space, envs.action_space, args)
        policy.init(*policy_args)
        policy = policy.to(args.device)
        policy.watch(log)
        policy.set_env_ref(envs)

        # Setup algo
        algo.set_get_policy(self.get_policy, policy_args)
        algo.set_env_ref(envs)
        #import ipdb; ipdb.set_trace()
        algo.init(policy, args)

        # Setup storage buffer
        storage = algo.get_storage_buffer(policy, envs, args)
        for ik, get_shape in alg_env_settings.include_info_keys:
            storage.add_info_key(ik, get_shape(envs))
        storage.to(args.device)
        storage.init_storage(envs.reset())
        storage.set_traj_done_callback(algo.on_traj_finished)

        simple_env = ['Sine-v0', 'SCurve-v0', 'Dalmatian-v0', 'Triangle-v0', 'Triangle-v2', 'Rectangle-v0', 'Rectangle-v1']

        runner = self._get_runner_cls(algo, policy)(
            envs, storage, policy, log, env_interface, checkpointer, args, algo
        )
        
        return runner

    def _get_runner_cls(self, algo, policy):
        return Runner

    def import_add(self):
        """
        Needed for ray training.
        """
        pass

    def setup(self, config):
        """
        Only called during ray training.
        """
        self.import_add()
        self.ray_runner = self.create_runner(config, ray_create=True)
        if self.ray_runner is None:
            return
        self.ray_runner.setup()
        self.ray_args = self.ray_runner.args

        if not self.ray_args.ray_debug:
            self.ray_runner.log.disable_print()

    def step(self):
        """
        Only called during ray training
        """
        updater_log_vals = self.ray_runner.training_iter(self.training_iteration)
        if (self.training_iteration + 1) % self.ray_args.log_interval == 0:
            log_dict = self.ray_runner.log_vals(
                updater_log_vals, self.training_iteration
            )
        if (self.training_iteration + 1) % self.ray_args.save_interval == 0:
            self.ray_runner.save(self.training_iteration)
        if (self.training_iteration + 1) % self.ray_args.eval_interval == 0:
            self.ray_runner.eval(self.training_iteration)

        return log_dict
