from rlf.envs.bit_flip import BIT_FLIP_ID
from gym.envs.registration import register

register(
        id=BIT_FLIP_ID,
        entry_point='tests.dev.her.bit_flip_env:BitFlipEnv',
        )
