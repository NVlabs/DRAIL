from rlf.rl.runner import Runner

class MultiRunner(Runner):
    def __init__(self, runners):
        self.runners = runners

    def training_iter(self, update_iter):
        train_log_vals = {}
        for runner in self.runners:
            lv = runner.training_iter(update_iter)
            train_log_vals.update(lv)
        return train_log_vals

    def setup(self):
        for runner in self.runners:
            runner.setup()

    def log_vals(self, updater_log_vals, update_iter):
        log_vals = {}
        for runner in self.runners:
            lv = runner.training_iter(update_iter)
            log_vals.update(lv)
        return log_vals

    def save(self, update_iter):
        for runner in self.runners:
            runner.save()

    def eval(self, update_iter):
        for runner in self.runners:
            runner.eval()

    def close(self):
        for runner in self.runners:
            runner.close()

    def resume(self):
        for runner in self.runners:
            step = runner.resume()
        # returning the last is assumed to be okay
        return step

    def should_load_from_checkpoint(self):
        all_load = []
        for runner in self.runners:
            all_load.append(runner.should_load_from_checkpoint())

        #TODO: Do some check that they are all the same
        return all_load[0]

    def full_eval(self, create_traj_saver_fn):
        eval_result = []
        for runner in self.runners:
            eval_result.append(runner.full_eval(create_traj_saver_fn))
        return eval_result

    def load_from_checkpoint(self):
        for runner in self.runners:
            runner.load_from_checkpoint()
