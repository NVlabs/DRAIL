from dowel import LogOutput
from dowel.tabular_input import TabularInput
import wandb
from rlf.exp_mgr import config_mgr

class WbOutput(LogOutput):
    def __init__(self, log_interval, args):
        wb_proj_name = config_mgr.get_prop('proj_name')
        wb_entity = config_mgr.get_prop('wb_entity')

        if args.prefix.count('-') >= 4:
            # Remove the seed and random ID info.
            parts = args.prefix.split('-')
            group_id = '-'.join([*parts[:2], *parts[4:]])
        else:
            group_id = None

        wandb.init(project=wb_proj_name, name=args.prefix,
                entity=wb_entity, group=group_id)
        wandb.config.update(args)

        self.log_dict = {}
        self.env_steps = None
        self.log_interval = log_interval

    @property
    def types_accepted(self):
        return (TabularInput,)

    def record(self, data, prefix=''):
        if not isinstance(data, TabularInput):
            raise ValueError('Unacceptable type')

        for key, value in data.as_dict.items():
            if key == 'TotalEnvSteps':
                self.env_steps = value
                continue
            self.log_dict[key] = value

    def dump(self, step=None):
        tmp_log_dict = self.log_dict
        self.log_dict = {}
        if step % self.log_interval != 0:
            return
        wandb.log(tmp_log_dict, step=self.env_steps)
