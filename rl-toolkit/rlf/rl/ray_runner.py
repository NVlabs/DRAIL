from ray import tune
import ray
import rlf.rl.utils as rutils
import pickle
from rlf.run_settings import RunSettings

class RunSettingsTrainable(tune.Trainable):
    run_settings: RunSettings = None

    def _setup(self, config):
        run_settings = pickle.loads(config['run_settings'])
        self.update_i = 0
        run_settings.import_add()
        self.runner = run_settings.create_runner(config)
        self.runner.setup()
        self.args = self.runner.args

        if not self.args.ray_debug:
            self.runner.log.disable_print()

    def _train(self):
        updater_log_vals = self.runner.training_iter(self.training_iteration)
        if (self.training_iteration+1) % self.args.log_interval == 0:
            log_dict = self.runner.log_vals(updater_log_vals, self.training_iteration)
        if (self.training_iteration+1) % self.args.save_interval == 0:
            self.runner.save()
        if (self.training_iteration+1) % self.args.eval_interval == 0:
            self.runner.eval()

        return log_dict




