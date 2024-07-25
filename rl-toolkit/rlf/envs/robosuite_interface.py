from rlf.envs.env_interface import EnvInterface, register_env_interface
from rlf.args import str2bool
import numpy as np
import gym
try:
    import robosuite
    from robosuite.wrappers import GymWrapper
    from robosuite.wrappers import DemoSamplerWrapper
except ImportError:
    pass


class RoboSuiteWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def seed(self, s):
        pass

    def render(self, mode='rgb_array'):
        rndr = self.env.env.sim.render(camera_name="frontview", width=128, height=128)
        rndr = np.flip(rndr)
        return rndr



class RoboSuiteControlInterface(EnvInterface):
    def env_trans_fn(self, env, set_eval):

        if self.args.rs_demo_wrapper:
            env = DemoSamplerWrapper(
                    env,
                    demo_path=self.args.demo_path,
                    need_xml=True,
                    num_traj=-1,
                    sampling_schemes=["uniform", "random"],
                    scheme_ratios=[0.9, 0.1],
                    )
            # For some reason this is reassigned
            env.action_spec = env.env.action_spec

        # Only render to camera when using as an evaluation environment.
        env = GymWrapper(env)
        env.reward_range = None
        env.metadata = None
        env.spec = None

        return RoboSuiteWrapper(env)

    def get_add_args(self, parser):
        parser.add_argument('--rs-reward-shaping', type=str2bool, default=True)
        parser.add_argument('--rs-demo-path')
        parser.add_argument('--rs-demo-wrapper', action='store_true',
                default=False)

    def create_from_id(self, env_id):
        _, task = env_id.split(".")
        # Env interface will do the actual job of creating the environment.
        env = robosuite.make(task, has_offscreen_renderer=set_eval,
                    has_renderer=False,
                    use_object_obs=True,
                    reward_shaping=self.args.rs_reward_shaping,
                    use_camera_obs=set_eval)
        return env



register_env_interface("^rs\.", RoboSuiteControlInterface)
