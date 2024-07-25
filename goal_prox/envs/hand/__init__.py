from gym.envs.registration import register
from goal_prox.envs.gym_hand import GymHandInterface
from rlf.envs.env_interface import register_env_interface


register(
    id="HandReachCustom-v0",
    entry_point="goal_prox.envs.hand.reach:HandReachEnv",
    kwargs={"reward_type": "sparse"},
    max_episode_steps=50,
)


register(
        id='CustomHandManipulateBlockRotateZ-v0',
        entry_point='goal_prox.envs.hand.manipulate:HandBlockEnv',
        kwargs={'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'sparse'},
        max_episode_steps=50,
    )

register(
        id='CustomHandManipulateBlockRotateZ-v2',
        entry_point='goal_prox.envs.hand.manipulate_v2:HandBlockEnv',
        kwargs={'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense'},
        max_episode_steps=50,
    )

register_env_interface("HandReachCustom-v0", GymHandInterface)
register_env_interface("CustomHandManipulateBlockRotateZ-v0", GymHandInterface)
register_env_interface("CustomHandManipulateBlockRotateZ-v2", GymHandInterface)