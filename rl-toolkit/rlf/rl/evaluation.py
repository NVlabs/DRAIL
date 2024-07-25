import os
import os.path as osp
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from rlf.exp_mgr.viz_utils import save_mp4
from rlf.il.traj_mgr import TrajSaver
from rlf.policies.base_policy import get_empty_step_info
from rlf.rl import utils
from rlf.rl.envs import get_vec_normalize, make_vec_envs
from tqdm import tqdm
import wandb
import time

def eval_print(
    env_interface,
    args,
    alg_env_settings,
    policy,
    vec_norm,
    total_num_steps,
    mode,
    eval_envs,
    log,
    final,
):
    print("Evaluating " + mode)
    args.evaluation_mode = True
    #ret_info, eval_envs, goal_achieved, eval_step_list
    eval_info, eval_envs, goal_achieved, _, eval_step_list  = evaluate(
        args,
        alg_env_settings,
        policy,
        vec_norm,
        env_interface,
        total_num_steps,
        mode,
        eval_envs,
        log,
        None,
        final,
    )

    log.log_vals(
        {"eval_%s_%s" % (mode, k): np.mean(v) for k, v in eval_info.items()},
        total_num_steps,
    )
    
    args.evaluation_mode = False
    return eval_envs, goal_achieved, eval_step_list


def train_eval(
    envs,
    alg_env_settings,
    policy,
    args,
    log,
    total_num_steps,
    env_interface,
    train_eval_envs,
    final,
    algo=None
):

    vec_norm = get_vec_normalize(envs)

    train_eval_envs, goal_achieved, eval_step_list = eval_print(
        env_interface,
        args,
        alg_env_settings,
        policy,
        vec_norm,
        total_num_steps,
        "train",
        train_eval_envs,
        log,
        final
    )

    return train_eval_envs, goal_achieved, eval_step_list


def full_eval(
    envs,
    policy,
    log,
    checkpointer,
    env_interface,
    args,
    alg_env_settings,
    create_traj_saver_fn,
    vec_norm,
    train_eval_envs
):
    
    args.evaluation_mode = True
    ret_info, envs, goal_achieved, goal_distance, eval_step_list = evaluate(
        args,
        alg_env_settings,
        policy,
        vec_norm,
        env_interface,
        0,
        "final",
        #envs,
        train_eval_envs,
        log,
        create_traj_saver_fn,
    )
    args.evaluation_mode = False
    envs.close()

    return ret_info, goal_achieved, goal_distance, eval_step_list

def evaluate(
    args,
    alg_env_settings,
    policy,
    true_vec_norm,
    env_interface,
    num_steps,
    mode,
    eval_envs,
    log,
    create_traj_saver_fn,
    final=False
):
    simple_env = ['Sine-v0', 'Sine-v1', 'SCurve-v0', 'Dalmatian-v0', 'Triangle-v0', 'Triangle-v2', 'Rectangle-v0', 'Rectangle-v1']

    if args.alg != 'dp' and args.clip_actions:
        ac_tensor = utils.ac_space_to_tensor(policy.action_space)
    
    if args.eval_num_processes is None:
        num_processes = args.num_processes
    else:
        num_processes = args.eval_num_processes

    if final:
        print("***** final evaluating *****")
        args.num_eval = 1000
   
    if eval_envs is None: #or (args.eval_only and args.env_name in simple_env):
        args.force_multi_proc = False
        # rlf.rl.envs.VecPyTorch
        origin_env, eval_envs = make_vec_envs(
            args.env_name,
            args.seed + num_steps,
            num_processes,
            args.gamma,
            args.device,
            True,
            env_interface,
            args,
            alg_env_settings,
            set_eval=True,
        )
 
        # if args.eval_only and args.env_name in simple_env:
        #     eval_envs = origin_env
    
    if args.env_name not in simple_env:
        assert get_vec_normalize(eval_envs) is None, "Norm is manually applied"

    if true_vec_norm is not None:
        obfilt = true_vec_norm._obfilt
    else:

        def obfilt(x, update):
            return x

    eval_episode_rewards = []
    eval_def_stats = defaultdict(list)
    ep_stats = defaultdict(list)

    obs = eval_envs.reset()
    #obs = state_action[:,0]

    hidden_states = {}
    for k, dim in policy.get_storage_hidden_states().items():
        hidden_states[k] = torch.zeros(num_processes, dim).to(args.device)
    eval_masks = torch.zeros(num_processes, 1, device=args.device)

    frames = []
    infos = None

    policy.eval()
    if args.eval_save and create_traj_saver_fn is not None:
        traj_saver = create_traj_saver_fn(
            osp.join(args.traj_dir, args.env_name, args.prefix)
        )
    else:
        assert not args.eval_save, (
            "Cannot save evaluation without ",
            "specifying the eval saver creator function",
        )

    total_num_eval = num_processes * args.num_eval
    
    # Measure the number of episodes completed
    pbar = tqdm(total=total_num_eval)
    evaluated_episode_count = 0
    n_succs = 0
    n_fails = 0
    succ_frames = []
    fail_frames = []

    if args.render_succ_fails and args.eval_num_processes > 1:
        raise ValueError(
            """
                Can only render successes and failures when the number of
                processes is 1.
                """
        )

    if args.num_render is None or args.num_render > 0:

        frames.extend(
            get_render_frames(
                eval_envs,
                env_interface,
                None,
                None,
                None,
                None,
                None,
                args,
                evaluated_episode_count,
            )
        )

    is_succ = False
    flag = 0
    count_flag = False
    step_num = 0 # the number counting time steps
    goal_achieved = []
    goal_distance = []
    eval_step_list = []
    if args.env_name == 'maze2d-medium-v2':
        fig, axs = plt.subplots(1, 2, figsize=(28, 3))
    initial_point = obs
    if args.env_name in simple_env:
        x = torch.Tensor()
        y = torch.Tensor()
        true_y_s = torch.Tensor()
    while evaluated_episode_count < total_num_eval:
        #tmp = time.time()

        step_info = get_empty_step_info()
        with torch.no_grad():
            act_obs = obfilt(utils.ob_to_np(obs), update=False)
            act_obs = utils.ob_to_tensor(act_obs, args.device)
            #import ipdb; ipdb.set_trace()
            ac_info = policy.get_action(
                utils.get_def_obs(act_obs),
                utils.get_other_obs(obs),
                hidden_states,
                eval_masks,
                step_info,
            )
            
            hidden_states = ac_info.hxs

            if args.alg != 'dp' and args.clip_actions:
                ac_info.clip_action(*ac_tensor)

        # Observe reward and next obs
        next_obs,  _, done, infos = eval_envs.step(ac_info.take_action)

        if args.env_name in simple_env:
            #import ipdb; ipdb.set_trace()
            if args.eval_only:
                x = torch.cat((x, obs.cpu()), dim=0)
            else:
                x = torch.cat((x, obs.cpu()), dim=0)
            y = torch.cat((y, ac_info.take_action), dim=0)

        if args.eval_save:
            finished_count = traj_saver.collect(
                obs, next_obs, done, ac_info.take_action, infos
            )
        else:
            finished_count = sum([int(d) for d in done])
            #finished_count = int(infos[0]["goal_achieved"])
        #print("loop %d if args.eval_save: %d" %(evaluated_episode_count, time.time()-tmp))
        #tmp = time.time()
        pbar.update(finished_count)
        evaluated_episode_count += finished_count

        cur_frame = None

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=args.device,
        )

        should_render = (args.num_render) is None or (
            evaluated_episode_count < args.num_render
        )
        if args.render_succ_fails:
            should_render = n_succs < args.num_render or n_fails < args.num_render
        #print("loop %d args.render_succ_fails: %d" %(evaluated_episode_count, time.time()-tmp))
        #tmp = time.time()
        if should_render:
            if args.env_name == 'maze2d-medium-v2' and flag <= 0:
                pass
            else:
                frames.extend(
                    get_render_frames(
                        eval_envs,
                        env_interface,
                        obs,
                        next_obs,
                        ac_info.take_action,
                        eval_masks,
                        infos,
                        args,
                        evaluated_episode_count,
                    )
                )
        obs = next_obs
        
        step_log_vals = utils.agg_ep_log_stats(infos, ac_info.extra)
        
        for k, v in step_log_vals.items():
            ep_stats[k].extend(v)
        #print("loop %d for k, v in step_log_vals.items(): %d" %(evaluated_episode_count, time.time()-tmp))
        #tmp = time.time()
        if args.env_name == 'maze2d-medium-v2':
            if count_flag:
                flag  = flag - 1
            
            if is_succ == False:
                step_num += 1
                if args.env_name == 'maze2d-medium-v2':
                    is_succ = infos[0]["goal_achieved"]
                else:
                    is_succ = infos[0]["ep_found_goal"]
                if is_succ:
                    flag = 6
                    count_flag = True
        #print("loop %d if count_flag: %d" %(evaluated_episode_count, time.time()-tmp))
        #tmp = time.time()
        if finished_count == 1:

            if args.env_name == 'maze2d-medium-v2':
                #is_succ = step_log_vals["ep_found_goal"][0]
                goal_achieved.append(is_succ)
                goal_distance.append(infos[0]["goal_distance"])
                next_ob =  next_obs.detach().cpu().numpy().squeeze()
                initial = initial_point.detach().cpu().numpy().squeeze()

                eval_step_list.append(step_num)
                is_succ = False
                flag = 0
                count_flag = False
                step_num = 0 # number counting time-steps return to zero

                if is_succ:
                    axs[0].scatter(initial[4], initial[5], color='red', edgecolor='white')
                    axs[0].scatter(next_ob[4], next_ob[5], color='blue', edgecolor='white')
                    axs[0].plot([initial[4], next_ob[4]], [initial[5], next_ob[5]], color='black')
                    axs[0].title.set_text('succ')
                else:
                    axs[1].scatter(initial[4], initial[5], color='red', edgecolor='white')
                    axs[1].scatter(next_ob[4], next_ob[5], color='blue', edgecolor='white')
                    axs[1].plot([initial[4], next_ob[4]], [initial[5], next_ob[5]],  color='black')
                    axs[1].title.set_text('fail')
                plt.savefig('goal.png')
                plt.close()  

                if "ep_success" in step_log_vals and args.render_succ_fails:
                    is_succ = step_log_vals["ep_success"][0]
                    if is_succ == 1.0:
                        if n_succs < args.num_render:
                            succ_frames.extend(frames)
                        n_succs += 1
                    else:
                        if n_fails < args.num_render:
                            fail_frames.extend(frames)
                        n_fails += 1

            if args.env_name not in simple_env and args.eval_only:
                save_frames(frames, 'each', evaluated_episode_count, args)
                frames = []
    
    if args.env_name in simple_env:
        data = torch.load(args.traj_load_path)
        expert_obs = np.squeeze(data['obs'].numpy())
        expert_actions = np.squeeze(data['actions'].numpy())

        plt.figure()
        plt.scatter(expert_obs, expert_actions, label='Ground Truth')
        plt.scatter(x, y, label='Predicted')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        env_name = args.env_name.lower().replace("-v0", "")
        if args.eval_only:
            plt.savefig(args.load_file.split('/')[4]+ '_eval_result.png')
        else:
            plt.savefig(args.prefix + '_result_' + args.env_name.lower().replace("-v0", "") + '.png')
        log.log_image("result", args.prefix + '_result_' + env_name + '.png', num_steps)
        print("x.shape:", x.shape)
        if args.rollout_agent:
            name = args.load_file.split('/')[-2]
            print("name:", name)
            #np.savez(env_name + '_' + args.alg + '_.npz', state=x.reshape(-1), action=y.reshape(-1))
            np.savez(name + '_' + args.alg + '_.npz', state=x.reshape(-1), action=y.reshape(-1))


    #tmp = time.time()
    pbar.close()
    info = {}
    if args.eval_save:
        traj_saver.save()

    ret_info = {}
    print(" Evaluation using %i episodes:" % len(ep_stats["r"]))

    for k, v in ep_stats.items():
        print(" - %s: %.5f" % (k, np.mean(v)))
        ret_info[k] = np.mean(v)
    
    if args.env_name == 'maze2d-medium-v2':
        succ_rate = np.sum(goal_achieved) / args.num_eval
        ret_info['goal_completion'] = succ_rate
        #print("timestep:", sum(eval_step_list) / len(eval_step_list))
        ret_info['time_step'] = sum(eval_step_list) / len(eval_step_list)

    if args.render_succ_fails:
        # Render the success and failures to two separate files.
        save_frames(succ_frames, "succ_" + mode, num_steps, args)
        save_frames(fail_frames, "fail_" + mode, num_steps, args)
    else:
        save_file = save_frames(frames, mode, num_steps, args)      

    # Switch policy back to train mode
    policy.train()

    return ret_info, eval_envs, goal_achieved, goal_distance, eval_step_list


def save_frames(frames, mode, num_steps, args):
    if not osp.exists(args.vid_dir):
        os.makedirs(args.vid_dir)

    add = ""
    if args.load_file != "":
        add = args.load_file.split("/")[-2]
        add += "_"

    save_name = "%s%s_%s" % (add, utils.human_format_int(num_steps), mode)

    save_dir = osp.join(args.vid_dir, args.env_name, args.prefix)

    fps = args.vid_fps

    if len(frames) > 0:
        save_mp4(frames, save_dir, save_name, fps=args.vid_fps, no_frame_drop=True)
        return osp.join(save_dir, save_name)
    return None


def get_render_frames(
    eval_envs,
    env_interface,
    obs,
    next_obs,
    action,
    masks,
    infos,
    args,
    evaluated_episode_count,
):
    add_kwargs = {}
    if args.render_metric:
        add_kwargs = {}
        if obs is not None:
            add_kwargs = {
                "obs": utils.ob_to_cpu(obs),
                "action": action.cpu(),
                "next_obs": utils.ob_to_cpu(next_obs),
                "info": infos,
                "next_mask": masks.cpu(),
            }

    try:
        cur_frame = eval_envs.render(**env_interface.get_render_args(), **add_kwargs)
    except EOFError as e:
        print("This problem can likely be fixed by setting --eval-num-processes 1")
        raise e

    if not isinstance(cur_frame, list):
        cur_frame = [cur_frame]
    return cur_frame
