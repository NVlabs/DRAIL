import numpy as np
import torch

from goal_prox.method.goal_traj_dataset import GoalTrajDataset


class GoalGAILTrajDataset(GoalTrajDataset):
    def __init__(self, load_path, args):
        self.args = args
        super().__init__(load_path)
        self.future_p = 1.0
        self.d = args.device

    def _gen_data(self, trajs):
        data = []
        for states, _ in trajs:
            traj = []
            T = len(states)
            for i in range(T - 1):
                traj.append({"ob": states[i], "next_ob": states[i + 1], "done": 0})
            traj[-1]["done"] = 1.0
            data.append(traj)
        return data

    def get_generator(self, num_batch, batch_size, relabel_ob, is_reached):
        self.relabel_ob = relabel_ob
        self.is_reached = is_reached
        for _ in range(num_batch):
            obs, next_obs, dones = self.sample_tensors(batch_size)
            yield {
                "state": obs,
                "next_state": next_obs,
                "done": dones,
            }

    def sample_tensors(self, sample_size):
        traj_idxs = np.random.randint(0, self.n_trajs, size=sample_size)
        trans_idxs = [
            np.random.randint(0, len(self.data[traj_idx])) for traj_idx in traj_idxs
        ]
        obs = []
        next_obs = []
        dones = []
        for i in range(sample_size):
            trans = self.data[traj_idxs[i]][trans_idxs[i]]
            ob = trans["ob"]
            next_ob = trans["next_ob"]
            done = trans["done"]

            # HER future sampling
            replace_goal = np.random.uniform() < self.future_p
            if replace_goal:
                future_idx = np.random.randint(
                    trans_idxs[i], len(self.data[traj_idxs[i]])
                )
                future_ob = self.data[traj_idxs[i]][future_idx]["next_ob"]
                ob = self.relabel_ob(ob, future_ob)
                next_ob = self.relabel_ob(next_ob, future_ob)

            obs.append(ob)
            next_obs.append(next_ob)
            dones.append(0)

        obs = torch.stack(obs).to(self.d)
        next_obs = torch.stack(next_obs).to(self.d)
        dones = torch.tensor(dones, device=self.d, dtype=torch.float32)

        return obs, next_obs, dones
