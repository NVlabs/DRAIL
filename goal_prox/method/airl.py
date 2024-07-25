from goal_prox.method.value_traj_dataset import *
from rlf.algos.il.gail import GailDiscrim
import torch.nn.functional as F
import torch
import torch.nn as nn
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from functools import partial
from rlf.comps.ensemble import Ensemble
from rlf.args import str2bool
import rlf.il.utils as iutils
import goal_prox.method.utils as mutils
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque, defaultdict
from rlf.policies.base_policy import get_step_info
from rlf.rl.loggers import sanity_checker


class ProxAirl(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([ProxAirlDiscrim(get_discrim), agent_updater], 1)


class ProxAirlDiscrim(GailDiscrim):
    def init(self, policy, args):
        super().init(policy, args)
        self.failure_agent_trajs = deque(maxlen=args.exp_buff_size)
        self.success_agent_trajs = deque(maxlen=args.exp_buff_size)

        is_img_obs = len(rutils.get_obs_shape(self.policy.obs_space)) == 3

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        settings.include_info_keys.extend([
            ('ep_found_goal', lambda _: (1,)),
            ('final_obs', lambda env: rutils.get_obs_shape(env.observation_space))
            ])
        return settings

    def _create_discrim(self):
        return Ensemble(super()._create_discrim, self.args.pf_n_nets)

    def compute_pen(self, expert_states, expert_actions, agent_states, agent_actions):
        return 0

    def _compute_prox_loss(self, pred_d, data_batch):
        proximity = data_batch['prox'].to(self.args.device)

        n_ensembles = pred_d.shape[0]
        return F.mse_loss(pred_d.view(n_ensembles, -1), proximity.view(1, -1).repeat(n_ensembles, 1))

    def _compute_discrim_reward(self, storage, step, add_info):
        obsfilt = self.get_env_ob_filt()
        if self.args.pf_diff_reward:
            def get_use_state(idx, sub_final):
                state = self._trans_agent_state(storage.get_obs(idx)).clone()
                if sub_final:
                    masks = storage.masks[idx]
                    finished_episodes = [i for i in range(len(masks)) if masks[i] == 0.0]
                    add_inputs = {k: v[idx-1] for k,v in add_info.items()}
                    for i in finished_episodes:
                        # The final state has no normalization!
                        state[i] = self._norm_expert_state(
                                add_inputs['final_obs'][i], obsfilt)
                return state

            def get_action(idx, sub_final):
                if not self.args.action_input:
                    return None
                if idx == len(storage.actions):
                    # Infer the action
                    idx_state = storage.get_obs(idx)
                    step_info = get_step_info(0, 0, 0, self.args)
                    ac_info = self.policy.get_action(rutils.get_def_obs(idx_state),
                            rutils.get_other_obs(idx_state),
                            storage.get_hidden_state(idx), storage.get_masks(idx),
                            step_info)
                    if self.args.clip_actions:
                        ac_info.clip_action(*self.ac_tensor)
                    actions = ac_info.action
                else:
                    actions = storage.actions[idx]
                if sub_final:
                    masks = storage.masks[idx]
                    finished_episodes = [i for i in range(len(masks)) if masks[i] == 0.0]
                    for i in finished_episodes:
                        actions[i] = torch.zeros(actions[i].shape).to( self.args.device)
                return actions

            cur_state = get_use_state(step, False)
            cur_action = get_action(step, False)

            next_masks = storage.masks[step+1]
            next_state = get_use_state(step+1, True)
            next_action = get_action(step+1, True)

            def get_prox(s, a):
                if a is not None:
                    a = rutils.get_ac_repr(self.policy.action_space, a)
                prox = self._compute_disc_val(s, a).mean(0)
                if self.args.pf_clip:
                    prox = torch.clamp(prox, 0.0, 1.0)
                return prox

            def get_prox_uncert(s, a):
                if a is not None:
                    a = rutils.get_ac_repr(self.policy.action_space, a)
                return self._compute_disc_val(s, a).std(0)

            cur_prox = get_prox(cur_state, cur_action)
            next_prox = get_prox(next_state, next_action)

            diff_prox_reward = (next_prox - cur_prox)
            final_prox_reward = next_prox * (1.0 - next_masks)

            uncert_pen = 0
            if self.args.pf_uncert_scale != 0 and self.args.pf_n_nets > 1:
                cur_uncert = get_prox_uncert(cur_state, cur_action)
                next_uncert = get_prox_uncert(next_state, next_action)
                uncert = torch.max(cur_uncert, next_uncert)
                uncert_pen = self.args.pf_uncert_scale * uncert

            reward = (diff_prox_reward + final_prox_reward - uncert_pen) * self.args.pf_reward_scale
            return reward
        else:
            state = rutils.get_def_obs(storage.get_obs(step))
            action = storage.actions[step]
            action = rutils.get_ac_repr(self.action_space, action)
            d_val = self._compute_disc_val(state, action)
            prox = d_val.mean(0)
            return prox

    def _compute_expert_loss(self, expert_d, expert_batch):
        return self._compute_prox_loss(expert_d, expert_batch)

    def _compute_agent_loss(self, agent_d, agent_batch):
        return self._compute_prox_loss(agent_d, agent_batch)

    def _get_traj_dataset(self, traj_load_path):
        if self.args.dmode == 'exp':
            use_fn = exp_discounted
        elif self.args.dmode == 'linear':
            use_fn = linear_discounted
        elif self.args.dmode == 'big':
            use_fn = partial(big_discounted, start_val=self.args.start_delta)
        elif self.args.dmode == 'one':
            use_fn = lambda T, t, delta: 1
        else:
            raise ValueError('Must specify discounting mode')
        self.compute_prox_fn = partial(use_fn, delta=self.args.pf_delta)
        return ValueTrajDataset(traj_load_path, self.compute_prox_fn, self.args)

    def get_add_args(self, parser):
        super().get_add_args(parser)

        parser.add_argument('--pf-traj-sampler', type=str2bool, default=False)
        parser.add_argument('--pf-with-success', type=str2bool, default=False)
        parser.add_argument('--exp-buff-size', type=int, default=4096)
        parser.add_argument('--exp-sample-size', type=int, default=4096)
        parser.add_argument('--exp-succ-scale', type=int, default=1)

        parser.add_argument('--pf-diff-reward', type=str2bool, default=False)
        parser.add_argument('--pf-uncert-scale', type=float, default=0.0)
        parser.add_argument('--pf-reward-scale', type=float, default=1.0)

        parser.add_argument('--dmode', type=str, default='one')
        parser.add_argument('--start-delta', type=float, default=1.0)
        parser.add_argument('--pf-delta', type=float, default=0.02)
        parser.add_argument('--pf-n-nets', type=int, default=1)
        parser.add_argument('--pf-clip', type=str2bool, default=False)

    def _get_sampler(self, storage):
        if self.args.pf_traj_sampler:
            take_count = self.args.exp_sample_size
            if len(self.failure_agent_trajs) < take_count:
                return None, None

            success_trajs = iutils.mix_data(self.success_agent_trajs, self.expert_dataset,
                    self.args.exp_succ_scale * take_count, 0.5)
            success_sampler = BatchSampler(SubsetRandomSampler(range(take_count)),
                    self.args.traj_batch_size, drop_last=True)
            success_trajs = iutils.convert_list_dict(success_trajs,
                    self.args.device)

            failure_trajs = self.failure_agent_trajs
            if len(self.failure_agent_trajs) > take_count:
                failure_trajs = np.random.choice(failure_trajs,
                        take_count, replace=False)
            failure_sampler = BatchSampler(SubsetRandomSampler(range(take_count)),
                    self.args.traj_batch_size, drop_last=True)
            failure_trajs = iutils.convert_list_dict(failure_trajs,
                    self.args.device)
            self.success_trajs = success_trajs
            self.failure_trajs = failure_trajs
            return success_sampler, failure_sampler
        else:
            agent_experience = storage.get_generator(None,
                                                     mini_batch_size=self.expert_train_loader.batch_size)
            return self.expert_train_loader, agent_experience

    def _trans_batches(self, expert_batch, agent_batch):
        if self.args.pf_traj_sampler:
            success_agent_batch = iutils.select_idx_from_dict(expert_batch,
                                                            self.success_trajs)
            failure_agent_batch = iutils.select_idx_from_dict(agent_batch,
                                                            self.failure_trajs)
            return success_agent_batch, failure_agent_batch
        else:
            agent_batch['prox'] = torch.zeros(agent_batch['state'].shape[0])
            return expert_batch, agent_batch

    def _trans_agent_state(self, state, other_state=None):
        if not self.args.pf_traj_sampler or self.args.gail_state_norm:
            return super()._trans_agent_state(state, other_state)
        assert other_state is None
        return rutils.get_def_obs(state, 'raw_obs')


    def on_traj_finished(self, trajs):
        super().on_traj_finished(trajs)
        if not self.args.pf_traj_sampler:
            return

        def _get_traj_tuples(actions, masks, raw_obs, final_state,
                             end_t, was_success):
            traj_actions = actions[:end_t]
            traj_masks = masks[:end_t]
            traj_raw_obs = raw_obs[:end_t]

            traj_raw_obs = torch.cat([traj_raw_obs, final_state[:,end_t-1]])
            all_actions = traj_actions.clone()
            all_actions = torch.cat([all_actions, torch.zeros(1,
                *traj_actions.shape[1:]).to(self.args.device)], dim=0)

            T = len(traj_raw_obs)
            if was_success:
                prox_target = torch.tensor([self.compute_prox_fn(T, t + 1) for t in range(T)])
            else:
                prox_target = torch.zeros(T)

            for j in range(len(traj_raw_obs)):
                yield {
                    'state': traj_raw_obs[j],
                    'prox': prox_target[j],
                    'action': all_actions[j],
                    }

        obs, obs_add, actions, masks, add_data, rewards = iutils.traj_to_tensor(trajs,
                                                                       self.args.device)
        n_trajs = len(trajs)
        final_state = add_data['final_obs'].unsqueeze(1)
        if not self.args.gail_state_norm:
            obs = obs_add['raw_obs']
        else:
            obsfilt = self.get_env_ob_filt()
            final_state = self._norm_expert_state(final_state, obsfilt)

        is_success, end_t = mutils.get_success(add_data, masks)
        if not self.args.pf_with_success:
            is_success = [False for _ in range(len(is_success))]

        for i in range(n_trajs):
            add_traj_tuples = list(_get_traj_tuples(actions[i], masks[i], obs[i],
                                                    final_state[i], end_t[i],
                                                    is_success[i]))
            if is_success[i]:
                use_traj_store = self.success_agent_trajs
            else:
                use_traj_store = self.failure_agent_trajs

            use_traj_store.extend(add_traj_tuples)
