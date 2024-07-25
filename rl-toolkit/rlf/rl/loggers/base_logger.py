import datetime
import os
import os.path as osp
import pipes
import random
import string
import sys
import time
from collections import defaultdict, deque
from typing import Any, Callable

import numpy as np
import torch
from rlf.exp_mgr import config_mgr
from six.moves import shlex_quote


class BaseLogger(object):
    def __init__(self, print_all=False):
        self._print_all = print_all

    def init(self, args, mod_prefix=lambda x: x):
        """
        - mod_prefix: Function that takes as input the prefix and returns the
          transformed prefix.
        """
        self.is_debug_mode = args.prefix == "debug"
        self._create_prefix(args, mod_prefix)

        print("Smooth len is %i" % args.log_smooth_len)

        self._step_log_info = defaultdict(lambda: deque(maxlen=args.log_smooth_len))

        if not self.is_debug_mode:
            self.save_run_info(args)
        else:
            print("In debug mode")
        self.is_printing = True
        self.prev_steps = 0
        self.start = None
        self.args = args
        self._collected_vals = defaultdict(list)

    def disable_print(self):
        self.is_printing = False

    def get_config(self):
        return self.args

    def save_run_info(self, args):
        log_dir = osp.join(args.log_dir, args.env_name, self.prefix)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        # cmd
        train_cmd = "python3 main.py " + " ".join(
            [pipes.quote(s) for s in sys.argv[1:]]
        )
        with open(osp.join(log_dir, "cmd.txt"), "a+") as f:
            f.write(train_cmd)

        # git diff
        print("Save git commit and diff to {}/git.txt".format(log_dir))
        cmds = [
            "echo `git rev-parse HEAD` >> {}".format(
                shlex_quote(osp.join(log_dir, "git.txt"))
            ),
            "git diff >> {}".format(shlex_quote(osp.join(log_dir, "git.txt"))),
        ]
        os.system("\n".join(cmds))

        args_lines = "Date and Time:\n"
        args_lines += time.strftime("%d/%m/%Y\n")
        args_lines += time.strftime("%H:%M:%S\n\n")
        arg_dict = args.__dict__
        for k in sorted(arg_dict.keys()):
            args_lines += "{}: {}\n".format(k, arg_dict[k])

        with open(osp.join(log_dir, "args.txt"), "w") as f:
            f.write(args_lines)

    def backup(self, args, global_step):
        log_dir = osp.join(args.log_dir, args.env_name, args.prefix)
        model_dir = osp.join(args.save_dir, args.env_name, args.prefix)
        vid_dir = osp.join(args.vid_dir, args.env_name, args.prefix)

        log_base_dir = log_dir.rsplit("/", 1)[0]
        model_base_dir = model_dir.rsplit("/", 1)[0]
        vid_base_dir = vid_dir.rsplit("/", 1)[0]
        proj_name = config_mgr.get_prop("proj_name")
        sync_host = config_mgr.get_prop("sync_host")
        sync_user = config_mgr.get_prop("sync_user")
        sync_port = config_mgr.get_prop("sync_port")
        cmds = [
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                sync_port, sync_user, sync_host, proj_name, log_dir
            ),
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                sync_port, sync_user, sync_host, proj_name, model_dir
            ),
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                sync_port, sync_user, sync_host, proj_name, vid_dir
            ),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                sync_port, log_dir, sync_user, sync_host, proj_name, log_base_dir
            ),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                sync_port, model_dir, sync_user, sync_host, proj_name, model_base_dir
            ),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                sync_port, vid_dir, sync_user, sync_host, proj_name, vid_base_dir
            ),
        ]
        os.system("\n".join(cmds))
        print("\n" + "*" * 50)
        print("*" * 5 + " backup at global step {}".format(global_step))
        print("*" * 50 + "\n")
        print("")

    def collect_step_info(self, step_log_info):
        for k in step_log_info:
            self._step_log_info[k].extend(step_log_info[k])

    def _get_env_id(self, args):
        upper_case = [c for c in args.env_name if c.isupper()]
        if len(upper_case) == 0:
            return "".join([word[0] for word in args.env_name.split(".")])
        else:
            return "".join(upper_case)

    def _create_prefix(self, args, mod_prefix):
        assert args.prefix is not None and args.prefix != "", "Must specify a prefix"
        d = datetime.datetime.today()
        date_id = "%i%i" % (d.month, d.day)
        env_id = self._get_env_id(args)

        chars = [x for x in string.ascii_uppercase + string.digits]
        rnd_id = np.random.RandomState().choice(chars, 2)
        rnd_id = "".join(rnd_id)

        before = "%s-%s-%s-%s-" % (date_id, env_id, args.seed, rnd_id)

        before = mod_prefix(before)

        if args.prefix != "debug" and args.prefix != "NONE":
            self.prefix = before + args.prefix
            print("Assigning full prefix %s" % self.prefix)
        else:
            self.prefix = args.prefix

    def set_prefix(self, args, setup_log_dirs=[]):
        args.prefix = self.prefix
        if setup_log_dirs:
            self.setup_log_dirs(args, setup_log_dirs)

    def setup_log_dirs(self, args, log_keys):
        for log_key in log_keys:
            new_dir = osp.join(getattr(args, log_key), args.prefix)
            setattr(args, log_key, new_dir)
            if not osp.exists(new_dir):
                os.makedirs(new_dir)

    def start_interval_log(self):
        """Old functionality, really we want to measure the average over the last
        time since logged, not over the single training step.
        """
        if self.start is None:
            self.start = time.time()

    def collect(self, val_name: str, val: Any) -> None:
        """Useful when collecting a bunch of logging variables over a loop but
        only want to report the final average.
        """
        self._collected_vals[val_name].append(val)

    def get_collect(self, val_name: str, list_idx: int) -> Any:
        """Gets a variable being collected."""
        return self._collected_vals[val_name][list_idx]

    def log_vals(self, key_vals, step_count, histo=False):
        """Log key value pairs to whatever interface. Also logs the collected
        values if there are any.
        """

        # Average the collected data
        def avg_data(x):
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x).detach().mean().item()
            else:
                return np.mean(x)

        collected_data = {k: avg_data(v) for k, v in self._collected_vals.items()}

        self._internal_log_vals({**key_vals, **collected_data}, step_count, histo)
        self._collected_vals = defaultdict(list)

    def _internal_log_vals(self, key_vals, step_count, histo):
        pass

    def log_video(self, video_file, step_count, fps):
        pass

    def watch_model(self, model, **kwargs):
        """
        - model (torch.nn.Module) the set of parameters to watch
        """
        pass

    def interval_log(
        self, num_updates, total_num_steps, episode_count, updater_log_vals, args, histo=False
    ):
        """
        Printed FPS is all inclusive of updates, evaluations, logging and everything.
        This is NOT the environment FPS.
        """
        end = time.time()

        fps = int((total_num_steps - self.prev_steps) / (end - self.start))
        self.prev_steps = total_num_steps
        self.start = time.time()
        num_eps = len(self._step_log_info.get("r", []))
        rewards = self._step_log_info.get("r", [0])

        log_stat_vals = {}
        for k, v in self._step_log_info.items():
            log_stat_vals["avg_" + k] = np.mean(v)
            log_stat_vals["min_" + k] = np.min(v)
            log_stat_vals["max_" + k] = np.max(v)

        def should_print(x):
            return "_pr_" in x

        log_dat = {
            **updater_log_vals,
            **log_stat_vals,
        }

        if self.is_printing:
            print(
                f"Updates {num_updates}, Steps {total_num_steps}, Episodes {episode_count}, FPS {fps}"
            )
            if args.num_steps != 0:
                print(
                    f"Over the last {num_eps} episodes:\n"
                    f"mean/median reward {np.mean(rewards):.2f}/{np.median(rewards)}\n"
                    f"min/max {np.min(rewards):.2f}/{np.max(rewards):.2f}"
                )

            # Print log values from the updater if requested.
            for k, v in log_dat.items():
                if should_print(k) or self._print_all:
                    print(f"    - {k}: {v}")

            # Print a new line to separate loggin lines and keep things clean.
            print("")
            print("")

        # Log additional core training metrics
        log_dat["fps"] = fps
        log_dat["episodes"] = episode_count
        log_dat["updates"] = num_updates

        self.log_vals(log_dat, total_num_steps, histo)
        return log_dat

    def log_image(self, k, img_file, step_count):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
