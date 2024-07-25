import os.path as osp
import os
from six.moves import shlex_quote
from rlf.rl import utils
import sys
import pipes
import time
import numpy as np
import random
import datetime
import string
import copy
from rlf.exp_mgr import config_mgr
from rlf.rl.loggers.base_logger import BaseLogger

from collections import deque, defaultdict



class TbLogger(BaseLogger):
    def __init__(self, tb_log_dir=None):
        super().__init__()
        self.tb_log_dir = tb_log_dir

    def init(self, args):
        super().init(args)
        if self.tb_log_dir is None:
            self.tb_log_dir = args.log_dir
        self.writer = self._create_writer(args, self.tb_log_dir)

    def _create_writer(self, args, log_dir):
        import ipdb; ipdb.set_trace()
        from tensorboardX import SummaryWriter
        rnd_id = ''.join(random.sample(string.ascii_uppercase + string.digits, k=4))
        log_dir = osp.join(self.tb_log_dir, args.env_name, args.prefix + '-' + rnd_id)
        #writer = SummaryWriter(log_dir)
        writer = SummaryWriter(log_dir, filename_suffix=f'_{run_id}')

        return writer

    def _internal_log_vals(self, key_vals, step_count):
        for k, v in key_vals.items():
            self.writer.add_scalar('data/' + k, v, step_count)
        self.writer.close()

    def close(self):
        self.writer.close()
