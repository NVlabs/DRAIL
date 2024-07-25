import numpy as np
import random
import torch
import rlf.rl.utils as rutils
import seaborn as sns

class DebugViz(object):
    def __init__(self, save_dir, args):
        self.save_dir = save_dir
        self.args = args

    def add(self, batches):
        pass

    def plot(self, iter_count, plot_names, plot_funcs):
        pass

    def reset(self):
        pass


PLOT_COUNT = 32
class LineDebugViz(DebugViz):
    def __init__(self, save_dir, dataset, args):
        super().__init__(save_dir, args)
        if args.traj_frac == 1.0:
            raise ValueError(('Must specify a fraction of the trajectories',
                    'is there can be a holdout set'))
        idxs = dataset.holdout_idxs
        self.trajs = [dataset.trajs[i] for i in idxs]
        self.plot_count = PLOT_COUNT

    def _plot_traj(self, traj, pf):
        states = traj[0]
        actions = traj[1]
        for i in range(len(states)):
            if self.args.action_input:
                use_actions = actions[i].to(self.args.device)
            else:
                use_actions = None

            yield pf(states[i].to(self.args.device), use_actions)


    def plot(self, iter_count, plot_names, plot_funcs):
        with torch.no_grad():
            all_traj_proxs = []
            use_trajs = random.sample(self.trajs, self.plot_count)
            pf = plot_funcs['prox']
            for traj in use_trajs:
                proxs = torch.stack(list(self._plot_traj(traj, pf)))
                all_traj_proxs.append(proxs)

        max_size = max([x.shape[0] for x in all_traj_proxs])
        proxs = torch.zeros(len(all_traj_proxs), max_size)
        for i, traj_proxs in enumerate(all_traj_proxs):
            proxs[i, :len(traj_proxs)] = traj_proxs.view(-1).cpu()

        sns.heatmap(proxs.numpy(), linewidth=0.5)
        rutils.plt_save(self.args.save_dir, f"dem_heat_{iter_count}.png")
