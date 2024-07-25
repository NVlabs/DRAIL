import os
import os.path as osp
from functools import partial

import numpy as np
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from rlf.algos.base_algo import BaseAlgo
from rlf.algos.il.bc import BehavioralCloning
from rlf.policies.random_policy import RandomPolicy
from rlf.rl.envs import make_vec_envs_easy
from rlf.rl.model import Flatten
from rlf.storage.rollout_storage import RolloutStorage
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm


class InvFunc(nn.Module):
    def __init__(self, get_state_enc, obs_shape, action_size, hidden_dim=64):
        super().__init__()
        self.is_img = len(obs_shape) == 3

        if self.is_img:
            n = obs_shape[1]
            m = obs_shape[2]
            image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
            self.head = nn.Sequential(
                nn.Conv2d(2 * obs_shape[0], 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                Flatten(),
                nn.Linear(image_embedding_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_size),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(2 * obs_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_size),
            )

    def forward(self, state_1, state_2):
        if self.is_img:
            x = torch.cat([state_1, state_2], dim=1)
        else:
            x = torch.cat([state_1, state_2], dim=-1)

        tmp = self.head(x)
        return tmp


def select_batch(trans_idx, dataset, device, ob_shape):
    """
    - trans_idx (list(int)): indices to select from dataset.
    - dataset (list(dict)): contains a list of dictionaries with the state
      action data.
    """
    use_state_0 = []
    use_state_1 = []
    use_action = []
    for i in trans_idx:
        use_state_0.append(dataset[i]["s0"])
        use_state_1.append(dataset[i]["s1"])
        use_action.append(dataset[i]["action"])
    use_state_0 = torch.stack(use_state_0).to(device).view(-1, *ob_shape)
    use_state_1 = torch.stack(use_state_1).to(device).view(-1, *ob_shape)
    true_action = torch.stack(use_action).to(device).squeeze(1)
    return use_state_0, use_state_1, true_action


class BehavioralCloningFromObs(BehavioralCloning):
    def init(self, policy, args):
        if args.bco_alpha != 0:
            # Adjust the amount of online experience based on args.
            args.num_env_steps = args.bco_alpha_size * args.bco_alpha
            args.num_steps = args.bco_alpha_size // args.num_processes
            print(f"Adjusted # steps to {args.num_steps}")
            print(f"Adjusted # env interactions to {args.num_env_steps}")

        super().init(policy, args)

        if self._arg("lr_env_steps") is None and args.bco_alpha != 0:
            # Adjust the learning rate decay steps based on the number of BC
            # updates performed.
            bc_updates = super().get_num_updates()
            bco_full_updates = self.get_num_updates() + 1
            # We perform a full BC update for each "BCO update". "BCO updates"
            # come from the initial training and for each update from online
            # experience determined according to alpha.
            self.lr_updates = bc_updates * bco_full_updates
            print(f"Adjusted # lr updates to {self.lr_updates}")

        get_state_enc = partial(
            self.policy.get_base_net_fn, rutils.get_obs_shape(self.policy.obs_space)
        )
        self.inv_func = InvFunc(
            get_state_enc,
            rutils.get_obs_shape(self.policy.obs_space),
            self.action_dim,
        )
        self.inv_func = self.inv_func.to(self.args.device)
        self.inv_opt = optim.Adam(self.inv_func.parameters(), lr=self.args.bco_inv_lr)

    def set_env_ref(self, envs):
        super().set_env_ref(envs)
        self.use_envs = envs

    def first_train(self, log, eval_policy, env_interface):
        """
        Gathers the random experience and trains the inverse model on it.
        """
        n_steps = self.args.bco_expl_steps // self.args.num_processes
        base_data_dir = "data/traj/bco"
        if not osp.exists(base_data_dir):
            os.makedirs(base_data_dir)

        loaded_traj = None
        if self.args.bco_expl_load is not None:
            load_path = osp.join(base_data_dir, self.args.bco_expl_load)
            if osp.exists(load_path) and not self.args.bco_expl_refresh:
                loaded_traj = torch.load(load_path)
                states = loaded_traj["states"]
                actions = loaded_traj["actions"]
                dones = loaded_traj["dones"]
                print(f"Loaded expl trajectories from {load_path}")

        if loaded_traj is None:
            policy = RandomPolicy()
            policy.init(
                self.use_envs.observation_space, self.use_envs.action_space, self.args
            )
            rutils.pstart_sep()
            print("Collecting exploration experience")
            states = []
            actions = []
            state = rutils.get_def_obs(self.use_envs.reset())
            states.extend(state)
            dones = [True]
            for _ in tqdm(range(n_steps)):
                ac_info = policy.get_action(state, None, None, None, None)
                state, reward, done, info = self.use_envs.step(ac_info.take_action)
                state = rutils.get_def_obs(state)
                actions.extend(ac_info.action)
                dones.extend(done)
                states.extend(state)
            rutils.pend_sep()
            self.use_envs.reset()

        if self.args.bco_expl_load is not None and loaded_traj is None:
            # Save the data.
            torch.save(
                {
                    "states": states,
                    "actions": actions,
                    "dones": dones,
                },
                load_path,
            )
            print(f"Saved data to {load_path}")

        if self.args.bco_inv_load is not None:
            self.inv_func.load_state_dict(torch.load(self.args.bco_inv_load))

        self._update_all(states, actions, dones)

    def _train_inv_func(self, trans_sampler, dataset):
        infer_ac_losses = []
        for i in tqdm(range(self.args.bco_inv_epochs)):
            for trans_idx in trans_sampler:
                use_state_0, use_state_1, true_action = select_batch(
                    trans_idx,
                    dataset,
                    self.args.device,
                    rutils.get_obs_shape(self.policy.obs_space),
                )
                pred_action = self.inv_func(use_state_0, use_state_1)
                loss = autils.compute_ac_loss(
                    pred_action,
                    true_action.view(-1, self.action_dim),
                    self.policy.action_space,
                )
                infer_ac_losses.append(loss.item())

                self.inv_opt.zero_grad()
                loss.backward()
                self.inv_opt.step()
        return infer_ac_losses

    def _infer_inv_accuracy(self, trans_sampler, dataset):
        total_count = 0
        num_correct = 0
        with torch.no_grad():
            for trans_idx in trans_sampler:
                use_state_0, use_state_1, true_action = select_batch(
                    trans_idx,
                    dataset,
                    self.args.device,
                    rutils.get_obs_shape(self.policy.obs_space),
                )
                pred_action = self.inv_func(use_state_0, use_state_1)
                pred_class = torch.argmax(pred_action, dim=-1)
                num_correct += (pred_class == true_action.view(-1)).float().sum()
                total_count += float(use_state_0.shape[0])
        return 100.0 * (num_correct / total_count)

    def _update_all(self, states, actions, dones):
        """
        - states (list[N+1])
        - masks (list[N+1])
        - actions (list[N])
        Performs a complete update of the model by following these steps:
            1. Train inverse function with ground truth data provided.
            2. Infer actions in expert dataset
            3. Train BC
        """
        dataset = [
            {"s0": states[i], "s1": states[i + 1], "action": actions[i]}
            for i in range(len(actions))
            if not dones[i + 1]
        ]

        rutils.pstart_sep()
        print(f"BCO Update {self.update_i}/{self.args.bco_alpha}")
        print("---")

        print("Training inverse function")
        dataset_idxs = list(range(len(dataset)))
        np.random.shuffle(dataset_idxs)

        eval_len = int(len(dataset_idxs) * self.args.bco_inv_eval_holdout)
        if eval_len != 0.0:
            train_trans_sampler = BatchSampler(
                SubsetRandomSampler(dataset_idxs[:-eval_len]),
                self.args.bco_inv_batch_size,
                drop_last=False,
            )
            val_trans_sampler = BatchSampler(
                SubsetRandomSampler(dataset_idxs[-eval_len:]),
                self.args.bco_inv_batch_size,
                drop_last=False,
            )
        else:
            train_trans_sampler = BatchSampler(
                SubsetRandomSampler(dataset_idxs),
                self.args.bco_inv_batch_size,
                drop_last=False,
            )

        if self.args.bco_inv_load is None or self.update_i > 0:
            infer_ac_losses = self._train_inv_func(train_trans_sampler, dataset)
            rutils.plot_line(
                infer_ac_losses,
                f"ac_inv_loss_{self.update_i}.png",
                self.args.vid_dir,
                not self.args.no_wb,
                self.get_completed_update_steps(self.update_i),
            )
            if self.update_i == 0:
                # Only save the inverse model on the first epoch for debugging
                # purposes
                rutils.save_model(
                    self.inv_func, f"inv_func_{self.update_i}.pt", self.args
                )

        if eval_len != 0.0:
            if not isinstance(self.policy.action_space, spaces.Discrete):
                raise ValueError(
                    (
                        "Evaluating the holdout accuracy is only",
                        " supported for discrete action spaces right now",
                    )
                )
            accuracy = self._infer_inv_accuracy(val_trans_sampler, dataset)
            print("Inferred actions with %.2f accuracy" % accuracy)

        if isinstance(self.expert_dataset, torch.utils.data.Subset):
            s0 = self.expert_dataset.dataset.trajs["obs"].to(self.args.device).float()
            s1 = (
                self.expert_dataset.dataset.trajs["next_obs"]
                .to(self.args.device)
                .float()
            )
            dataset_device = self.expert_dataset.dataset.trajs["obs"].device
        else:
            s0 = self.expert_dataset.trajs["obs"].to(self.args.device).float()
            s1 = self.expert_dataset.trajs["next_obs"].to(self.args.device).float()
            dataset_device = self.expert_dataset.trajs["obs"].device

        # Perform inference on the expert states
        with torch.no_grad():
            pred_actions = self.inv_func(s0, s1).to(dataset_device)
            pred_actions = rutils.get_ac_compact(self.policy.action_space, pred_actions)
            if not self.args.bco_oracle_actions:
                if isinstance(self.expert_dataset, torch.utils.data.Subset):
                    self.expert_dataset.dataset.trajs["actions"] = pred_actions
                else:
                    self.expert_dataset.trajs["actions"] = pred_actions
        # Recreate the dataset for BC training so we can be sure it has the
        # most recent data.
        self._create_train_loader(self.args)

        print("Training Policy")
        self.full_train(self.update_i)
        self.update_i += 1
        rutils.pend_sep()

    def get_num_updates(self):
        if self.args.bco_alpha == 0:
            return 0
        return BaseAlgo.get_num_updates(self)

    def get_completed_update_steps(self, num_updates):
        return BaseAlgo.get_completed_update_steps(self, num_updates)

    def update(self, storage):
        masks = storage.masks.view(-1, 1)
        actions = storage.actions.view(-1, storage.actions.shape[-1])
        obs = storage.get_def_obs_seq()
        obs = obs.view(-1, *obs.shape[2:])
        # Update based on collected experience from environment
        dones = [(not bool(x)) for x in masks]
        self._update_all(obs, actions, dones)
        return {}

    def get_storage_buffer(self, policy, envs, args):
        # Rollout buffer to store the collected online experience
        return RolloutStorage(
            args.num_steps,
            args.num_processes,
            envs.observation_space,
            envs.action_space,
            args,
        )

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(
            "--bco-expl-steps",
            type=int,
            default=1000,
            help="""
                Number of random exploration steps.
                """,
        )

        # Inverse model learning arguments.
        parser.add_argument(
            "--bco-inv-lr",
            type=float,
            default=0.001,
            help="""
                The learning rate of the action inverse model
                """,
        )
        parser.add_argument(
            "--bco-inv-epochs",
            type=int,
            default=1,
            help="""
                The number of epochs when training the inverse model. This is
                used both for the alpha updates and the pre-alpha updates.
                """,
        )
        parser.add_argument(
            "--bco-inv-eval-holdout",
            type=float,
            default=0.0,
            help="""
                The fraction of data that should be withheld when training the
                inverse model and later used for evaluation.
                """,
        )
        parser.add_argument(
            "--bco-inv-batch-size",
            type=int,
            default=32,
            help="""
                The batch size for the inverse action model training.
                """,
        )

        # Online learning arguments.
        parser.add_argument(
            "--bco-alpha",
            type=int,
            default=0,
            help="""
                Number of online updates.
                """,
        )
        parser.add_argument(
            "--bco-alpha-size",
            type=int,
            default=1000,
            help="""
                Size of each online update.
                """,
        )
        parser.add_argument(
            "--bco-oracle-actions",
            action="store_true",
            help="""
                Use the ground truth expert actions rather than the inferred
                ones. FOR DEBUGGING ONLY. This does not use the learned inverse
                model.
                """,
        )

        # Saving files to speed things up arguments
        parser.add_argument(
            "--bco-expl-load",
            type=str,
            default=None,
            help="""
                File to load/save the random explorations. If the file exists
                the random explorations will be loaded from here. If it does
                not exist the random explorations will be saved here.
                """,
        )
        parser.add_argument(
            "--bco-expl-refresh",
            action="store_true",
            help="""
                Regenerate the random explorations even if the expl-load file
                is specified and exists.
                """,
        )
        parser.add_argument(
            "--bco-inv-load",
            type=str,
            default=None,
            help="""
                If specified the inverse model will be loaded from here and
                not trained on the random exploration phase. However, it
                **will** be trained on all subsequent alpha updates.
                """,
        )
