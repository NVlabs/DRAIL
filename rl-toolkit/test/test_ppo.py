import os.path as osp

import pytest
from rlf import run_policy
from rlf.algos import PPO, SAC
from rlf.policies import DistActorCritic, DistActorQ
from rlf.run_settings import RunSettings

NUM_ENV_SAMPLES = 1000
NUM_STEPS = 100
NUM_PROCS = 2


class PPORunSettings(RunSettings):
    def get_config_file(self):
        config_dir = osp.dirname(osp.realpath(__file__))
        return osp.join(config_dir, "config.yaml")

    def get_policy(self):
        return DistActorCritic()

    def get_algo(self):
        return PPO()


def test_cont_train():
    TEST_ENV = "Pendulum-v0"
    run_settings = PPORunSettings(
        f"--prefix 'ppo-test' --use-proper-time-limits --linear-lr-decay True --lr 3e-4 --entropy-coef 0 --num-env-steps {NUM_ENV_SAMPLES} --num-mini-batch 32 --num-epochs 10 --num-steps {NUM_STEPS} --env-name {TEST_ENV} --eval-interval -1 --log-smooth-len 10 --save-interval -1 --num-processes {NUM_PROCS} --cuda False"
    )
    run_policy(run_settings)


def test_disc_train():
    TEST_ENV = "Acrobot-v1"
    run_settings = PPORunSettings(
        f"--prefix 'ppo-test' --use-proper-time-limits --linear-lr-decay True --lr 3e-4 --entropy-coef 0 --num-env-steps {NUM_ENV_SAMPLES} --num-mini-batch 32 --num-epochs 10 --num-steps {NUM_STEPS} --env-name {TEST_ENV} --eval-interval -1 --log-smooth-len 10 --save-interval -1 --num-processes {NUM_PROCS} --cuda False"
    )
    run_policy(run_settings)
