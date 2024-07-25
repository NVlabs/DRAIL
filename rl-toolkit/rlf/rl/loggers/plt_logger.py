from rlf.rl.loggers.base_logger import BaseLogger
from collections import defaultdict
import rlf.rl.utils as rutils

class PltLogger(BaseLogger):
    def __init__(self, save_keys, x_name, y_names, titles):
        super().__init__()
        self.save_keys = save_keys
        self.x_name = x_name
        self.y_names = y_names
        self.titles = titles
        self.logged_vals = defaultdict(list)
        self.logged_steps = defaultdict(list)

    def init(self, args):
        self.args = args
        super().init(args)

    def log_vals(self, key_vals, step_count):
        for k, v in key_vals.items():
            self.logged_vals[k].append(v)
            self.logged_steps[k].append(step_count)

    def close(self):
        # Plot everything
        for k, y_name, title in zip(self.save_keys, self.y_names, self.titles):
            rutils.plot_line(self.logged_vals[k], f"final_{k}", self.args,
                    False, x_vals=self.logged_steps[k], x_name=self.x_name,
                    y_name=y_name, title=title)
