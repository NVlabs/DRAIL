from rlf.algos.il.base_irl import BaseIRLAlgo
import torch
import torch.nn as nn
from rlf.args import str2bool
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os.path as osp
import os
from collections import deque, defaultdict
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.comps.ensemble import Ensemble
from rlf.rl.model import InjectNet
from rlf.policies.base_policy import get_step_info
import rlf.algos.utils as autils
import rlf.policies.utils as putils
from rlf.exp_mgr.viz_utils import append_text_to_image

from functools import partial
import itertools

import rlf.rl.utils as rutils
import goal_prox.method.utils as mutils
import rlf.il.utils as iutils
from abc import ABC, abstractmethod
from rlf.rl.loggers import sanity_checker


def get_default_pf(n_layers, hidden_dim):
    modules = []
    #modules.append(nn.Linear(add_dim+input_dim, hidden_dim))
    #modules.append(nn.Tanh())

    for i in range(n_layers-1):
        modules.append(nn.Linear(hidden_dim, hidden_dim))
        modules.append(nn.Tanh())
    return nn.Sequential(
            *modules,
            nn.Linear(hidden_dim, 1))

def safe_get_action(actions, action_idx, args):
    if action_idx >= len(actions):
        return torch.zeros(actions[0].shape).to(args.device)
    return actions[action_idx]


class ProxFunc(BaseIRLAlgo, ABC):
    def __init__(self, get_pf=None, get_pf_base=None):
        super().__init__()
        if get_pf is None:
            get_pf = get_default_pf
        self.get_pf_base = None
        self.get_pf = get_pf

    def init(self, policy, args):
        super().init(policy, args)

        def create_prox_func():
            obsp = rutils.get_obs_shape(self.policy.obs_space)
            if self.get_pf_base is None:
                state_enc = self.policy.get_base_net_fn(obsp)
            else:
                state_enc = self.get_pf_base(obsp)

            in_dim = state_enc.output_shape[0]
            pf_head = self.get_pf(n_layers=self.args.pf_n_layers,
                    hidden_dim=self.args.pf_n_hidden)

            return InjectNet(state_enc.net, pf_head,
                    in_dim, self.args.pf_n_hidden,
                    rutils.get_ac_dim(self.policy.action_space),
                    args.action_input).to(args.device)

        self.prox_func = Ensemble(create_prox_func, self.args.pf_n_nets)
        self.opt = optim.Adam(self.prox_func.parameters(),
                              lr=self.args.prox_lr)

        self.model_save_dir = osp.join(
            args.save_dir, args.env_name, args.prefix)

        self.debug_viz = mutils.get_visualizer(args, policy,
                self.expert_dataset, args.pf_viz_type)
        self.start_proxs = []
        self.avg_proxs = []

        self.exp_buff_size = args.exp_buff_size
        if self.exp_buff_size == -1:
            self.exp_buff_size = mutils.get_default_buff_size(args)
        self.failure_agent_trajs = deque(maxlen=self.exp_buff_size)
        self.success_agent_trajs = deque(maxlen=self.exp_buff_size)

        is_img_obs = len(rutils.get_obs_shape(self.policy.obs_space)) == 3
        if is_img_obs and self.args.pf_state_norm:
            raise ValueError('Illegal to perform state normalization with images')
        self.use_raw_obs = not is_img_obs and args.normalize_env

        if self.args.pf_reward_norm:
            self.returns = None
            self.ret_rms = RunningMeanStd(shape=())

    def _get_prox_uncert(self, state, action):
        assert self.args.pf_n_nets > 1
        pval = self._get_prox_vals(state, action)
        return pval.std(0)

    def _get_prox(self, state, action, should_clip):
        state = self._preproc_pf_input(state)
        if action is not None:
            action = rutils.get_ac_repr(self.policy.action_space, action)
        pval = self.prox_func(state, action).mean(0)

        if should_clip:
            pval = torch.clamp(pval, 0.0, 1.0)
        return pval

    def _get_prox_vals(self, state, action):
        state = self._preproc_pf_input(state)
        if action is not None:
            action = rutils.get_ac_repr(self.policy.action_space, action)
        pval = self.prox_func(state, action)
        return pval
    
    def eval_disc(self, state, action):
        state = self._preproc_pf_input(state)
        if action is not None:
            action = rutils.get_ac_repr(self.policy.action_space, action)
        self.prox_func.eval()
        with torch.no_grad():
            pval = self.prox_func(state, action)
        pval = pval.mean()
        return pval

    def _get_reward(self, step, storage, add_info):
        with torch.no_grad():
            self.prox_func.eval()
            def get_use_state(idx, sub_final):
                state = storage.get_obs(idx)
                if self.use_raw_obs:
                    state = state['raw_obs']
                else:
                    state = rutils.get_def_obs(state)
                state = state.clone()
                if sub_final:
                    masks = storage.masks[idx]
                    finished_episodes = [i for i in range(len(masks)) if masks[i] == 0.0]
                    add_inputs = {k: v[idx-1] for k,v in add_info.items()}
                    for i in finished_episodes:
                        state[i] = add_inputs['final_obs'][i]
                return state

            def get_action(idx, sub_final):
                if idx == len(storage.actions):
                    # Infer the action
                    idx_state = storage.get_obs(idx)
                    step_info = get_step_info(0, 0, 0, self.args)
                    with torch.no_grad():
                        ac_info = self.policy.get_action(
                                rutils.get_def_obs(idx_state),
                                rutils.get_other_obs(idx_state),
                                storage.get_hidden_state(idx),
                                storage.get_masks(idx), step_info)
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
            if self.args.action_input:
                cur_action = get_action(step, False)
            else:
                cur_action = None

            next_masks = storage.masks[step+1]
            next_state = get_use_state(step+1, True)

            if self.args.action_input:
                next_action = get_action(step+1, True)
            else:
                next_action = None

            cur_prox = self._get_prox(cur_state, cur_action, self.args.pf_clip)
            next_prox = self._get_prox(next_state, next_action, self.args.pf_clip)
            constant_pen = 0
            uncert_pen = 0
            log_dict = {}

            if self.args.pf_uncert and self.args.pf_n_nets > 1:
                next_uncert = self._get_prox_uncert(next_state, next_action)
                uncert_pen = self.args.pf_uncert_scale * next_uncert

                log_dict.update({
                    'uncert_pen': uncert_pen,
                    'uncert': next_uncert,
                    })

            if self.args.pf_reward_type == 'reg':
                diff_prox_reward = (next_prox - cur_prox)
                final_prox_reward = next_prox * (1.0 - next_masks)
            elif self.args.pf_reward_type == 'nodiff':
                diff_prox_reward = next_prox
                final_prox_reward = next_prox * (1.0 - next_masks)
            elif self.args.pf_reward_type == 'nofinal':
                diff_prox_reward = (next_prox - cur_prox)
                final_prox_reward = torch.zeros(diff_prox_reward.shape).to(self.args.device)
            elif self.args.pf_reward_type == 'nofinal_airl':
                eps = 1e-20
                s = next_prox
                next_prox_ = (s + eps).log() - (1 - s + eps).log()
                s = cur_prox
                cur_prox_ = (s + eps).log() - (1 - s + eps).log()
                diff_prox_reward = (next_prox - cur_prox)
                final_prox_reward = torch.zeros(diff_prox_reward.shape).to(self.args.device)
            elif self.args.pf_reward_type == 'nofinal_pen':
                diff_prox_reward = (next_prox - cur_prox)
                final_prox_reward = torch.zeros(diff_prox_reward.shape).to(self.args.device)
                constant_pen = self.args.pf_constant_penalty
            elif self.args.pf_reward_type == 'none':
                diff_prox_reward = next_prox
                final_prox_reward = torch.zeros(diff_prox_reward.shape).to(self.args.device)
            elif self.args.pf_reward_type == 'pen':
                diff_prox_reward = torch.zeros(cur_prox.shape).to(self.args.device)
                final_prox_reward = torch.zeros(cur_prox.shape).to(self.args.device)
                constant_pen = self.args.pf_constant_penalty
            elif self.args.pf_reward_type == 'airl':
                eps = 1e-20
                s = next_prox
                diff_prox_reward = (s + eps).log() - (1 - s + eps).log()
                final_prox_reward = torch.zeros(diff_prox_reward.shape).to(self.args.device)
            else:
                raise ValueError("unrecongized reward type")
            reward = (diff_prox_reward + final_prox_reward - uncert_pen + constant_pen) * self.args.pf_reward_scale

            if self.args.pf_reward_norm:
                # Normalize reward
                if self.returns is None:
                    self.returns = reward.clone()
                self.returns = self.returns * storage.masks[step] * self.args.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
                reward = reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

            log_dict.update({
                    'diff_prox_reward': diff_prox_reward,
                    'final_prox_reward': final_prox_reward,
                    'prox_reward': diff_prox_reward + final_prox_reward,
                    })

            return reward, log_dict

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)

        settings.include_info_keys.extend([
            ('ep_found_goal', lambda _: (1,)),
            ('final_obs', lambda env: rutils.get_obs_shape(env.observation_space))
            ])
        settings.ret_raw_obs = True

        settings.mod_render_frames_fn = self.mod_render_frames
        return settings

    def mod_render_frames(self, frame, env_cur_obs, env_cur_action, env_cur_reward,
            env_next_obs, **kwargs):
        use_obs = rutils.get_def_obs(env_cur_obs)
        use_obs = torch.FloatTensor(use_obs).unsqueeze(0).to(self.args.device)
        # Infer the proximity
        prox = self._get_prox(use_obs, None, self.args.pf_clip)
        uncert = self._get_prox_uncert(use_obs, None)
        frame = append_text_to_image(frame, [
            "Prox: %.3f" % prox,
            "Uncert: %.3f" % uncert,
            "Reward: %.3f" % (env_cur_reward if env_cur_reward is not None else 0.0)
            ])
        return frame

    def first_train(self, log, eval_policy, env_interface):
        if self.args.pf_load_path is not None:
            self.prox_func.load_state_dict(torch.load(self.args.pf_load_path)['prox_func'])
            print('Loaded proximity function from %s' % self.args.pf_load_path)
            return

        losses = []

        # Train the proximity function from scratch
        rutils.pstart_sep()
        print('Pre-training proximity function')

        self.prox_func.train()

        for epoch_i in tqdm(range(self.args.pre_num_epochs)):
            epoch_losses = []
            for expert_batch in self.expert_train_loader:
                loss = self._prox_func_iter(expert_batch)
                epoch_losses.append(loss.item())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print('Epoch %i: Loss %.5f' % (epoch_i, avg_loss))

        plot_title = None
        if self.val_train_loader is not None:
            avg_val_loss = self._compute_val_loss()
            plot_title = 'Val Loss %.5f' % avg_val_loss

        # Save a figure of the loss curve

        self.debug_viz.plot(0, ["expert"], self._get_plot_funcs())

        rutils.pend_sep()

    def _compute_val_loss(self):
        val_losses = []
        with torch.no_grad():
            for val_expert_batch in self.val_train_loader:
                val_loss = self._prox_func_iter(val_expert_batch)
                val_losses.append(val_loss.item())
        return np.mean(val_losses)

    def _preproc_pf_input(self, states):
        if self.args.pf_state_norm:
            return (states - self.expert_stats['state'][0]) / (self.expert_stats['state'][1] + 1e-7)
        return states

    @abstractmethod
    def _prox_func_iter(self, data_batch):
        pass

    def should_use_failure(self):
        return True

    def _update_reward_func(self, storage):
        if not self.args.pf_with_agent:
            # Don't use agent experience to update the proximity function.
            return {}

        take_count = self.args.exp_sample_size

        if self.should_use_failure() and len(self.failure_agent_trajs) < take_count:
            # We don't have enough agent experience yet to update the proximity
            # function.
            return {}

        success_trajs = iutils.mix_data(self.success_agent_trajs, self.expert_dataset,
                self.args.exp_succ_scale * take_count, 0.5)
        success_sampler = BatchSampler(SubsetRandomSampler(range(take_count)),
                self.args.traj_batch_size, drop_last=True)

        success_trajs = iutils.convert_list_dict(success_trajs,
                self.args.device)

        if self.should_use_failure():
            failure_trajs = self.failure_agent_trajs
            if len(self.failure_agent_trajs) > take_count:
                failure_trajs = np.random.choice(failure_trajs,
                        take_count, replace=False)
            failure_sampler = BatchSampler(SubsetRandomSampler(range(take_count)),
                    self.args.traj_batch_size, drop_last=True)
            failure_trajs = iutils.convert_list_dict(failure_trajs,
                    self.args.device)
        else:
            failure_sampler = itertools.repeat({})

        log_vals = defaultdict(list)
        self.prox_func.train()
        for epoch_i in range(self.args.pf_num_epochs):
            for success_idx, failure_idx in zip(success_sampler, failure_sampler):
                viz_dict = {}
                combined_loss = 0.0

                success_agent_batch = iutils.select_idx_from_dict(success_idx,
                                                                success_trajs)
                viz_dict['success'] = success_agent_batch
                expert_loss = self._prox_func_iter(success_agent_batch)
                log_vals['expert_loss'].append(expert_loss.item())
                combined_loss += expert_loss

                if self.should_use_failure():
                    failure_agent_batch = iutils.select_idx_from_dict(failure_idx,
                                                                    failure_trajs)
                    agent_loss = self._prox_func_iter(failure_agent_batch)
                    log_vals['agent_loss'].append(agent_loss.item())
                    viz_dict['failure'] = failure_agent_batch
                    combined_loss += agent_loss

                    grad_pen = 0
                    if self.args.disc_grad_pen != 0:
                        grad_pen = self.args.disc_grad_pen * autils.wass_grad_pen(
                                success_agent_batch['state'],
                                success_agent_batch['actions'],
                                failure_agent_batch['state'],
                                failure_agent_batch['actions'],
                                self.args.action_input, self._get_prox_vals)
                    combined_loss += grad_pen

                self.debug_viz.add(viz_dict)

                self.opt.zero_grad()
                combined_loss.backward()
                self.opt.step()

                log_vals['combined_loss'].append(combined_loss.item())

        for k in log_vals:
            log_vals[k] = np.mean(log_vals[k])
        if self.val_train_loader is not None:
            log_vals['expert_val_loss'] = self._compute_val_loss()

        if self.update_i % self.args.pf_viz_interval == 0:
            self.debug_viz.plot(self.update_i, ['success', 'failure'],
                                self._get_plot_funcs())
        # Still clear the viz statistics, even if we did not log.
        self.debug_viz.reset()

        if len(self.avg_proxs) != 0:
            log_vals['avg_traj_prox'] = np.mean(self.avg_proxs)
        if len(self.start_proxs) != 0:
            log_vals['start_traj_proxs'] = np.mean(self.start_proxs)
        self.start_proxs = []
        self.avg_proxs = []

        return log_vals

    def _get_plot_funcs(self):
        plot_funcs = {
            "prox": partial(self._get_prox, should_clip=True),
        }
        if self.args.pf_n_nets > 1 and self.args.pf_uncert:
            plot_funcs['uncert'] = self._get_prox_uncert
            def r(state,action):
                return self._get_prox(state,action, should_clip=True) - self._get_prox_uncert(state,action)
            plot_funcs['reward'] = r
        return plot_funcs

    def compute_good_traj_prox(self, obs, actions):
        pass

    def compute_bad_traj_prox(self, obs, actions):
        pass

    def _get_traj_tuples(self, actions, masks, raw_obs, final_state,
                         end_t, was_success):
        traj_actions = actions[:end_t]
        traj_masks = masks[:end_t]
        traj_raw_obs = raw_obs[:end_t]

        traj_raw_obs = torch.cat([traj_raw_obs, final_state[:,end_t-1]])

        if was_success:
            prox_fn = self.compute_good_traj_prox
        else:
            prox_fn = self.compute_bad_traj_prox
        prox_target = prox_fn(traj_raw_obs, traj_actions)
        all_actions = traj_actions.clone()
        all_actions = torch.cat([all_actions, torch.zeros(1,
            *traj_actions.shape[1:]).to(self.args.device)], dim=0)

        with torch.no_grad():
            # Infer the proximities of this trajectory for debugging purposes.
            traj_proxs = self._get_prox(traj_raw_obs, all_actions, self.args.pf_clip)
            traj_proxs = traj_proxs.cpu().numpy()

            self.start_proxs.append(traj_proxs[0, 0])
            self.avg_proxs.append(np.mean(traj_proxs))

        for j in range(len(traj_raw_obs)):
            yield {
                'state': traj_raw_obs[j],
                'prox': prox_target[j],
                'actions': all_actions[j],
                }

    def on_traj_finished(self, trajs):
        super().on_traj_finished(trajs)
        obs, obs_add, actions, masks, add_data, rewards = iutils.traj_to_tensor(trajs,
                                                                       self.args.device)
        if not self.args.pf_with_agent:
            return

        n_trajs = len(trajs)
        if self.use_raw_obs:
            obs = obs_add['raw_obs']
        final_state = add_data['final_obs'].unsqueeze(1)

        is_success, end_t = mutils.get_success(add_data, masks)
        if not self.args.pf_with_success:
            is_success = [False for _ in range(len(is_success))]

        for i in range(n_trajs):
            add_traj_tuples = self._get_traj_tuples(actions[i], masks[i], obs[i],
                                                    final_state[i], end_t[i], is_success[i])

            if is_success[i]:
                use_traj_store = self.success_agent_trajs
            else:
                use_traj_store = self.failure_agent_trajs

            use_traj_store.extend(add_traj_tuples)

    def get_add_args(self, parser):
        super().get_add_args(parser)

        parser.add_argument('--no-wb', action='store_true')

        #########################################
        # New args
        parser.add_argument('--pf-with-agent', type=str2bool, default=True)
        parser.add_argument('--pf-with-success', type=str2bool, default=False)
        parser.add_argument('--pf-uncert', type=str2bool, default=True)
        parser.add_argument('--pf-uncert-scale', type=float, default=0.01)

        parser.add_argument('--pf-gw-gt', type=str2bool, default=False)
        parser.add_argument('--pf-viz-type', type=str, default=None)
        parser.add_argument('--pf-viz-interval', type=int, default=10)

        parser.add_argument('--pf-n-nets', type=int, default=5)
        parser.add_argument('--pf-n-layers', type=int, default=2)
        parser.add_argument('--pf-n-hidden', type=int, default=64)
        parser.add_argument('--pre-num-epochs', type=int, default=5)
        parser.add_argument('--pf-num-epochs', type=int, default=1)
        parser.add_argument('--pf-load-path', type=str, default=None)
        parser.add_argument('--prox-lr', type=float, default=0.001)

        parser.add_argument('--action-input', type=str2bool, default=False)
        parser.add_argument('--pf-state-norm', type=str2bool, default=False)
        parser.add_argument('--pf-reward-norm', type=str2bool, default=True)
        parser.add_argument('--pf-clip', type=str2bool, default=True)
        parser.add_argument('--disc-grad-pen', type=float, default=0.0)

        parser.add_argument('--exp-buff-size', type=int, default=10000)
        parser.add_argument('--exp-sample-size', type=int, default=128)
        parser.add_argument('--exp-succ-scale', type=int, default=1)

        parser.add_argument('--pf-reward-scale', type=float, default=1.0)
        parser.add_argument('--pf-reward-type', type=str, default='nofinal')
        parser.add_argument('--pf-constant-penalty', type=float, default=-0.005)

    def load(self, checkpointer):
        super().load_resume(checkpointer)
        self.prox_func.load_state_dict(checkpointer.get_key('prox_func'))

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        self.opt.load_state_dict(checkpointer.get_key('pf_opt'))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key('pf_opt', self.opt.state_dict())
        checkpointer.save_key('prox_func', self.prox_func.state_dict())
