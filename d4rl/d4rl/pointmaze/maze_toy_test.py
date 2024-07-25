# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random


WALL = 10
EMPTY = 11
GOAL = 12


def parse_maze(maze_str):
    lines = maze_str.strip().split("\\")
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == "#":
                maze_arr[w][h] = WALL
            elif tile == "G":
                maze_arr[w][h] = GOAL
            elif tile == " " or tile == "O" or tile == "0":
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError("Unknown tile type: %s" % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel("point_maze")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler"
    )
    default = mjcmodel.root.default()
    default.joint(damping=1, limited="false")
    default.geom(
        friction=".5 .1 .1",
        density="1000",
        margin="0.002",
        condim="1",
        contype="2",
        conaffinity="1",
    )

    asset = mjcmodel.root.asset()
    asset.texture(
        type="2d",
        name="groundplane",
        builtin="checker",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        width=100,
        height=100,
    )
    asset.texture(
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=".4 .6 .8",
        rgb2="0 0 0",
        width="800",
        height="800",
        mark="random",
        markrgb="1 1 1",
    )
    asset.material(name="groundplane", texture="groundplane", texrepeat="20 20")
    asset.material(name="wall", rgba=".7 .5 .3 1")
    asset.material(name="target", rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4", diffuse=".8 .8 .8", specular="0.1 0.1 0.1")
    visual.map(znear=0.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(
        name="ground",
        size="40 40 0.25",
        pos="0 0 -0.1",
        type="plane",
        contype=1,
        conaffinity=0,
        material="groundplane",
    )

    particle = worldbody.body(name="particle", pos=[1.2, 1.2, 0])
    particle.geom(
        name="particle_geom", type="sphere", size=0.1, rgba="0.0 0.0 1.0 0.0", contype=1
    )
    particle.site(
        name="particle_site", pos=[0.0, 0.0, 0], size=0.2, rgba="0.3 0.6 0.3 1"
    )
    particle.joint(name="ball_x", type="slide", pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name="ball_y", type="slide", pos=[0, 0, 0], axis=[0, 1, 0])

    worldbody.site(name="target_site", pos=[0.0, 0.0, 0], size=0.2, material="target")

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w, h] == WALL:
                worldbody.geom(
                    conaffinity=1,
                    type="box",
                    name="wall_%d_%d" % (w, h),
                    material="wall",
                    pos=[w + 1.0, h + 1.0, 0],
                    size=[0.5, 0.5, 0.2],
                )

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


LARGE_MAZE = (
    "############\\"
    + "#OOOO#OOOOO#\\"
    + "#O##O#O#O#O#\\"
    + "#OOOOOO#OOO#\\"
    + "#O####O###O#\\"
    + "#OO#O#OOOOO#\\"
    + "##O#O#O#O###\\"
    + "#OO#OOO#OGO#\\"
    + "############"
)

LARGE_MAZE_EVAL = (
    "############\\"
    + "#OO#OOO#OGO#\\"
    + "##O###O#O#O#\\"
    + "#OO#O#OOOOO#\\"
    + "#O##O#OO##O#\\"
    + "#OOOOOO#OOO#\\"
    + "#O##O#O#O###\\"
    + "#OOOO#OOOOO#\\"
    + "############"
)

MEDIUM_MAZE = (
    "########\\"
    + "#OO##OO#\\"
    + "#OO#OOO#\\"
    + "##OOO###\\"
    + "#OO#OOO#\\"
    + "#O#OO#O#\\"
    + "#OOO#OG#\\"
    + "########"
)

MEDIUM_MAZE_EVAL = (
    "########\\"
    + "#OOOOOG#\\"
    + "#O#O##O#\\"
    + "#OOOO#O#\\"
    + "###OO###\\"
    + "#OOOOOO#\\"
    + "#OO##OO#\\"
    + "########"
)

SMALL_MAZE = "######\\" + "#OOOO#\\" + "#O##O#\\" + "#OOOO#\\" + "######"

U_MAZE = "#####\\" + "#GOO#\\" + "###O#\\" + "#OOO#\\" + "#####"

U_MAZE_EVAL = "#####\\" + "#OOG#\\" + "#O###\\" + "#OOO#\\" + "#####"

OPEN = (
    "#######\\"
  + "#OOOOO#\\"
  + "#OOOOO#\\"
  + "#OOOOO#\\"
  + "#OOOOO#\\"
  + "#OOOOG#\\"
  + "#######"
)


class MazeToyTest(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(
        self, maze_spec=U_MAZE, reward_type="dense", reset_target=False, **kwargs
    ):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()

        self._target = np.array([0.0, 0.0])

        model = point_maze(maze_spec)

        # rand_num = random.random()
        # if rand_num < 1/3:
        #     self.goal_locations = [(1, 1)]
        # elif rand_num < 2/3:
        #     self.goal_locations = [(1, 3)]
        # else:
        #     self.goal_locations = [(1, 5)]
        self.goal_locations = [(2, 3)]
        #import ipdb; ipdb.set_trace()
        self.set_target(self.goal_locations[0])
        
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations

        # For different coverages
        self._initial_locations = self._target_locations = self.empty_and_goal_locations
        self.set_coverage(1.0)
        # self.set_noise_ratio(0.1, 0)
        self.set_noise_ratio(1.0, 0)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)


    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = self._get_obs()
        if self.reward_type == "sparse":
            #reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.15 else 0.0
            reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == "dense":
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError("Unknown reward type %s" % self.reward_type)
        # done = False

        # Add goal_achieved to info dict
        # goal_achieved = True if np.linalg.norm(ob[0:2] - self._target) <= 0.1 else False
        goal_achieved = True if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else False
        x = ob[0]
        y = ob[1]
        # fall = (x > 1.3) and (y > 1.3)
        # done = goal_achieved or fall
        done = goal_achieved
        return ob, reward, done, dict(goal_achieved=goal_achieved)

    def _get_obs(self):
        # Add target position to observation
        return np.concatenate(
            [self.sim.data.qpos, self.sim.data.qvel, self._target]
        ).ravel()

    def get_target(self):
        return self._target

    ##### target_location is not None, so this block does not work
    def set_target(self, target_location=None):
        if target_location is None:
            # idx = self.np_random.choice(len(self.empty_and_goal_locations))
            # reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            idx = self.np_random.choice(len(self._target_locations))
            reset_location = np.array(self._target_locations[idx]).astype(
                self.observation_space.dtype
            )
            target_location = (
                reset_location
                + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
                * self.goal_noise_ratio
            )
        self._target = target_location

    def set_marker(self):
        #import ipdb; ipdb.set_trace()
        self.data.site_xpos[self.model.site_name2id("target_site")] = np.array(
            [self._target[0] + 1, self._target[1] + 1, 0.0]
        )

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel,-5.0 , 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    ##### revise
    def reset_model(self):
        reset_location = np.array([5,3]).astype(
            self.observation_space.dtype
        )
        qpos = (
            reset_location
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            * self.noise_ratio
        )
        qvel = (
            self.init_qvel
            + self.np_random.randn(self.model.nv) * 0.1 * self.noise_ratio
        )

        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_model_(self, state):
        reset_location = np.array(state[4:6].cpu()).astype(self.observation_space.dtype)
        qpos = np.array(state[0:2].cpu())
        qvel = np.array(state[2:4].cpu())
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target(reset_location)
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = (
            self.model.stat.extent * 1.0
        )  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.lookat[
            0
        ] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        # self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        # self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis
        pass

    def set_coverage(self, coverage, seed=123):
        self.coverage = coverage
        rng = np.random.RandomState(seed)
        n = len(self.empty_and_goal_locations)

        self._initial_locations = rng.permutation(self.empty_and_goal_locations)[
            : int(n * coverage)
        ]
        self._target_locations = rng.permutation(self.empty_and_goal_locations)[
            : int(n * coverage)
        ]

    def set_noise_ratio(self, noise_ratio, goal_noise_ratio):
        self.noise_ratio = noise_ratio
        self.goal_noise_ratio = goal_noise_ratio

    def relabel_ob(self, ob_current, ob_future):
        import torch
        if isinstance(ob_current, torch.Tensor):
            return torch.cat([ob_current[:-2], ob_future[:2]])
        return np.concatenate([ob_current[:-2], ob_future[:2]]).ravel()

    def is_reached(self, ob):
        import torch
        if isinstance(ob, torch.Tensor):
            return 1.0 if np.linalg.norm(ob[0:2].cpu() - ob[-2:].cpu()) <= 0.5 else 0.0
        return 1.0 if np.linalg.norm(ob[0:2] - ob[-2:]) <= 0.5 else 0.0

