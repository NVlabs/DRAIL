"""
Stand alone script to generate the D4RL maze2d dataset in our format.
"""
import gym
import d4rl
import torch
import argparse
import os
import os.path as osp
import numpy as np
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model

from rlf.exp_mgr.viz_utils import save_mp4


ENV = "maze2d-medium-v1"
SAVE_DIR = "data/traj"


def reset_data():
    return {
        "obs": [],
        "next_obs": [],
        "actions": [],
        "done": [],
    }


def append_data(data, s, ns, a, done):
    data["obs"].append(s)
    data["next_obs"].append(ns)
    data["actions"].append(a)
    data["done"].append(done)


def extend_data(data, episode):
    data["obs"].extend(episode["obs"])
    data["next_obs"].extend(episode["next_obs"])
    data["actions"].extend(episode["actions"])
    data["done"].extend(episode["done"])


def npify(data):
    for k in data:
        if k == "dones":
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, default="maze2d-medium-v1", help="Maze type"
    )
    parser.add_argument("--coverage", type=float, default=1.0)
    parser.add_argument("--noise_ratio", type=float, default=1.0)
    parser.add_argument("--goal_noise_ratio", type=float, default=1.0)
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Num episodes to collect"
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_steps", type=int, default=75)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)
    env.seed(args.seed)
    env.reset_target = True
    env.set_coverage(args.coverage)
    env.set_noise_ratio(args.noise_ratio, args.goal_noise_ratio)

    np.random.seed(args.seed)

    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    episode = reset_data()
    ts = 0
    cnt_episodes = 0
    if args.record:
        frames = [env.render("rgb_array")]
    while cnt_episodes < args.num_episodes:
        position = s[0:2]
        velocity = s[2:4]
        act, done = controller.get_action(position, velocity, env._target)

        act = act + np.random.randn(*act.shape) * 0.5
        act = np.clip(act, -1.0, 1.0)

        if ts >= max_episode_steps:
            done = True

        ns, _, _, _ = env.step(act)
        append_data(episode, s, ns, act, done)

        if args.record:
            frames.append(env.render("rgb_array"))

        ts += 1
        if done:
            if args.record:
                # frames = np.stack(frames)
                save_mp4(frames, "./", "maze2d", fps=30, no_frame_drop=True)
                frames = []

            print(ts)
            s = env.reset()
            controller = waypoint_controller.WaypointController(maze)

            if args.min_steps < ts < max_episode_steps:
                extend_data(data, episode)
                cnt_episodes += 1
                if cnt_episodes % 100 == 0:
                    print("Episodes: ", cnt_episodes, ",  States", len(data["obs"]))
            ts = 0
            episode = reset_data()
        else:
            s = ns

    npify(data)

    save_name = args.env_name.replace("-", "_") + "_expert_dataset_%d_%d.pt" % (
        args.num_episodes,
        args.coverage * 100,
    )
    dones = data["done"]
    obs = data["obs"]
    next_obs = data["next_obs"]
    actions = data["actions"]
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(
        {
            "done": torch.FloatTensor(dones),
            "obs": torch.tensor(obs),
            "next_obs": torch.tensor(next_obs),
            "actions": torch.tensor(actions),
        },
        osp.join(SAVE_DIR, save_name),
    )
    print("Saved to ", save_name)
    print("Num episodes:", cnt_episodes)
    print("Num steps:", len(data["obs"]))


if __name__ == "__main__":
    main()
