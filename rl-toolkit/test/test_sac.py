import os.path as osp

import pytest
from rlf import run_policy
from rlf.algos import PPO, SAC
from rlf.policies import DistActorCritic, DistActorQ
from rlf.run_settings import RunSettings

NUM_ENV_SAMPLES = 1000
NUM_STEPS = 1


class SacRunSettings(RunSettings):
    def get_config_file(self):
        config_dir = osp.dirname(osp.realpath(__file__))
        return osp.join(config_dir, "config.yaml")

    def get_policy(self):
        return DistActorQ()

    def get_algo(self):
        return SAC()


def test_sac_cont_train():
    TEST_ENV = "Pendulum-v0"
    run_settings = SacRunSettings(
        f"--prefix 'sac-test' --use-proper-time-limits --linear-lr-decay True --lr 3e-4 --num-env-steps {NUM_ENV_SAMPLES} --num-steps {NUM_STEPS} --env-name {TEST_ENV} --eval-interval -1 --log-smooth-len 10 --save-interval -1 --num-processes 1 --cuda False --n-rnd-steps 10"
    )
    run_policy(run_settings)
