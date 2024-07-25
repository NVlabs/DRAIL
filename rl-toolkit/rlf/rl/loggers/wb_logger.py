import copy
import datetime
import os
import os.path as osp
import pipes
import random
import string
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
from rlf.exp_mgr import config_mgr
from rlf.rl import utils
from rlf.rl.loggers.base_logger import BaseLogger
from six.moves import shlex_quote
import matplotlib.pyplot as plt

# Does not necessarily have WB installed
try:
    import wandb
except:
    pass

# Does not necessarily have Ray installed
try:
    from ray.tune.integration.wandb import WandbLogger
    from ray.tune.logger import DEFAULT_LOGGERS
except:
    pass


def get_wb_ray_kwargs():
    return {"loggers": DEFAULT_LOGGERS + (WandbLogger,)}


def get_wb_ray_config(config):
    config["wandb"] = {
        "project": config_mgr.get_prop("proj_name"),
        "api_key": config_mgr.get_prop("wb_api_key"),
        "log_config": True,
    }
    return config


def get_wb_media(v):
    if isinstance(v, torch.Tensor) and len(v.shape) == 1 and v.shape[0] > 1:
        v = wandb.Histogram(v.numpy())
    if isinstance(v, np.ndarray) and len(v.shape) == 1 and v.shape[0] > 1:
        v = wandb.Histogram(v)

    return v


class WbLogger(BaseLogger):
    """
    Logger for logging to the weights and W&B online service.
    """

    def __init__(
        self,
        wb_proj_name=None,
        should_log_vids=False,
        wb_entity=None,
        skip_create_wb=False,
    ):
        """
        - wb_proj_name: (string) if None, will use the proj_name provided in
          the `config.yaml` file.
        """
        super().__init__()
        if wb_proj_name is None:
            wb_proj_name = config_mgr.get_prop("proj_name")
        if wb_entity is None:
            wb_entity = config_mgr.get_prop("wb_entity")
        self.wb_proj_name = wb_proj_name
        self.wb_entity = wb_entity
        self.should_log_vids = should_log_vids
        self.skip_create_wb = skip_create_wb

    def init(self, args, mod_prefix=lambda x: x):
        super().init(args, mod_prefix)
        if self.skip_create_wb:
            return
        self.wandb = self._create_wandb(args)

    def _internal_log_vals(self, key_vals, step_count, histo=False):
        if self.is_closed:
            return
        # try:
        if not histo:
            every_key_vals = {k: get_wb_media(v) for k, v in key_vals.items() if k[0] != '_'}
            wandb.log(every_key_vals, step=int(step_count))
            if "_reward_map" in key_vals:
                self.log_image("reward_map", key_vals["_reward_map"], int(step_count))
            if "_disc_val_map" in key_vals:
                self.log_image("disc_val_map", key_vals["_disc_val_map"], int(step_count))
        # else:
            # key_vals = {k: wandb.plot.histogram(
            #                     wandb.Table(data=v, columns=["rewards"]), 
            #                     "rewards", 
            #                     title=k
            #                 ) for k, v in key_vals.items() if k[0] == '_'}
            # for k, v in key_vals.items():
                # wandb.log({k: v}, step=step_count)
            # for k, v in key_vals.items():
            #     if k[0] == '_':
            #         plt.figure()
            #         plt.hist(np.asarray(v).squeeze(1), bins=20)
            #         plt.savefig(f"./data/hist/{k}.png")
            #         self.log_image(k, f"./data/hist/{k}.png", step_count)
                
        # except Exception as e:
        #    print(e)
        #    self.is_closed = True
        #    print("Wb logger was closed when trying to log")

    def watch_model(self, model, log_type="gradients", log_freq=1000, **kwargs):
        wandb.watch(model, log=log_type, log_freq=log_freq)

    def log_image(self, k, img_file, step_count):
        wandb.log({k: wandb.Image(img_file)}, step=step_count)

    def _create_wandb(self, args):
        args.prefix = self.prefix
        if self.prefix.count("-") >= 4:
            # Remove the seed and random ID info.
            parts = self.prefix.split("-")
            # group_id = "-".join([*parts[:2], *parts[4:]])
            group_id = self.prefix
        else:
            group_id = None

        self.run = wandb.init(
            project=self.wb_proj_name,
            name=self.prefix,
            entity=self.wb_entity,
            # group=group_id,
            reinit=True,
            config=args,
        )
        self.is_closed = False
        return wandb

    def get_config(self):
        return wandb.config

    def log_video(self, video_file, step_count, fps):
        if not self.should_log_vids:
            return
        wandb.log({"video": wandb.Video(video_file + ".mp4", fps=fps, format="mp4")}, step=step_count)

    def close(self):
        self.is_closed = True
        # self.run.finish()
        wandb.run.save()
