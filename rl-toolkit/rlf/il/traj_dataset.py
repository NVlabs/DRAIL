import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.il.il_dataset import ImitationLearningDataset, convert_to_tensors


class TrajDataset(ImitationLearningDataset):
    """
    See `rlf/il/il_dataset.py` for notes about the demonstration dataset
    format.
    """

    def __init__(self, load_path, transform_dem_dataset_fn=None):
        super().__init__(load_path, transform_dem_dataset_fn)
        trajs = torch.load(load_path)

        rutils.pstart_sep()
        self._setup(trajs)

        trajs = self._generate_trajectories(trajs)

        assert len(trajs) != 0, "No trajectories found to load!"

        self.n_trajs = len(trajs)
        print("Collected %i trajectories" % len(trajs))

        self.data = self._gen_data(trajs)
        self.traj_lens = [len(traj[0]) for traj in trajs]
        self.trajs = trajs
        self.holdout_idxs = []

        rutils.pend_sep()

    def get_num_trajs(self):
        return self.n_trajs

    def compute_split(self, traj_frac, rnd_seed):
        traj_count = int(len(self.trajs) * traj_frac)
        all_idxs = np.arange(0, len(self.trajs))

        rng = np.random.default_rng(rnd_seed)
        rng.shuffle(all_idxs)
        idxs = all_idxs[:traj_count]
        self.holdout_idxs = all_idxs[traj_count:]
        self.n_trajs = traj_count

        self.data = self._gen_data([self.trajs[i] for i in idxs])
        return self

    def _setup(self, trajs):
        """
        Initialization subclasses need to perform. Cannot perform
        initialization in __init__ as traj is not avaliable.
        """
        pass

    def viz(self, args):
        import seaborn as sns

        sns.distplot(self.traj_lens)
        rutils.plt_save(args.save_dir, args.env_name, args.prefix, "traj_len_dist.png")

    def get_expert_stats(self, device):
        # Compute statistics across the trajectories.
        all_obs = torch.cat([t[0] for t in self.trajs])
        all_actions = torch.cat([t[1] for t in self.trajs])

        self.state_mean = torch.mean(all_obs, dim=0)
        self.state_std = torch.std(all_obs, dim=0)
        self.action_mean = torch.mean(all_actions, dim=0)
        self.action_std = torch.std(all_actions, dim=0)

        return {
            "state": (self.state_mean.to(device), self.state_std.to(device)),
            "action": (self.action_mean.to(device), self.action_std.to(device)),
        }

    def __getitem__(self, i):
        return self.data[i]

    def _gen_data(self, trajs):
        """
        Can define in inhereted class to perform a custom transformation over
        the trajectories.
        """
        return trajs

    def should_terminate_traj(self, j, obs, next_obs, done, actions):
        return done[j]

    def _generate_trajectories(self, trajs):
        is_tensor_dict = not isinstance(trajs["obs"], torch.Tensor)
        if not is_tensor_dict:
            trajs = convert_to_tensors(trajs)

        # Get by trajectory instead of transition
        if is_tensor_dict:
            for name in ["obs", "next_obs"]:
                for k in trajs[name]:
                    trajs[name][k] = trajs["obs"][k].float()
            obs = rutils.transpose_dict_arr(trajs["obs"])
            next_obs = rutils.transpose_dict_arr(trajs["next_obs"])
        else:
            obs = trajs["obs"].float()
            next_obs = trajs["next_obs"].float()

        done = trajs["done"].float()
        actions = trajs["actions"].float()

        ret_trajs = []

        num_samples = done.shape[0]
        print("Collecting trajectories")
        start_j = 0
        j = 0
        while j < num_samples:
            if self.should_terminate_traj(j, obs, next_obs, done, actions):
                obs_seq = obs[start_j : j + 1]
                final_obs = next_obs[j]

                combined_obs = [*obs_seq, final_obs]
                # combined_obs = torch.cat([obs_seq, final_obs.view(1, *obs_dim)])

                ret_trajs.append((combined_obs, actions[start_j : j + 1]))
                # Move to where this episode ends
                while j < num_samples and not done[j]:
                    j += 1
                start_j = j + 1

            if j < num_samples and done[j]:
                start_j = j + 1

            j += 1

        for i in range(len(ret_trajs)):
            states, actions = ret_trajs[i]
            if is_tensor_dict:
                states = rutils.transpose_arr_dict(states)
            else:
                states = torch.stack(states, dim=0)
            ret_trajs[i] = (states, actions)

        ret_trajs = self._transform_dem_dataset_fn(ret_trajs, trajs)
        return ret_trajs

    def __len__(self):
        return len(self.data)
