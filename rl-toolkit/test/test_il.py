from rlf.algos import PPO, GAIL, GAIFO, BehavioralCloningFromObs, BehavioralCloning
from rlf import run_policy
from rlf.policies import DistActorCritic, BasicPolicy
import os.path as osp
import os
from rlf.run_settings import RunSettings
import pytest


class GeneralRunSettings(RunSettings):
    def __init__(self, algo_type, policy_type, args_str):
        super().__init__(args_str)
        self.algo_type = algo_type
        self.policy_type = policy_type

    def get_config_file(self):
        config_dir = osp.dirname(osp.realpath(__file__))
        return osp.join(config_dir, 'config.yaml')

    def get_policy(self):
        return self.policy_type()

    def get_algo(self):
        return self.algo_type()



def test_save_load_train():
    TEST_ENV = 'Pendulum-v0'
    EXPERT_NUM_ENV_SAMPLES = 400
    EXPERT_NUM_STEPS = 50
    EXPERT_NUM_PROCS = 2

    # Train expert
    run_settings = GeneralRunSettings(PPO, DistActorCritic, f"--prefix ppo-test --linear-lr-decay True --lr 3e-4 --entropy-coef 0 --num-env-steps {EXPERT_NUM_ENV_SAMPLES} --num-mini-batch 32 --num-epochs 10 --num-steps {EXPERT_NUM_STEPS} --env-name {TEST_ENV} --eval-interval -1 --log-smooth-len 10 --save-interval 1 --num-processes {EXPERT_NUM_PROCS} --cuda False")
    run_result = run_policy(run_settings)
    expected_file = f"./data/trained_models/{TEST_ENV}/{run_result.prefix}/model_4.pt"
    assert osp.exists(expected_file)

    # Evaluate expert and save dataset
    run_settings = GeneralRunSettings(PPO, DistActorCritic, f"--prefix ppo-test --env-name {TEST_ENV} --eval-only --cuda False --load-file {expected_file} --eval-save --eval-num-processes 1 --num-processes 1 --num-render 0")
    run_result = run_policy(run_settings)

    expected_trajs = f"./data/traj/{TEST_ENV}/{run_result.prefix}/trajs.pt"

    # Run GAIL
    run_settings = GeneralRunSettings(GAIL, DistActorCritic, f"--prefix gail-test --env-name {TEST_ENV} --cuda False --traj-load-path {expected_trajs} --num-env-steps {EXPERT_NUM_ENV_SAMPLES} --num-steps {EXPERT_NUM_STEPS} --num-processes {EXPERT_NUM_PROCS} --eval-interval -1 --save-interval -1")
    run_policy(run_settings)

    # Run GAIFO
    run_settings = GeneralRunSettings(GAIFO, DistActorCritic, f"--prefix gail-test --env-name {TEST_ENV} --cuda False --traj-load-path {expected_trajs} --num-env-steps {EXPERT_NUM_ENV_SAMPLES} --num-steps {EXPERT_NUM_STEPS} --num-processes {EXPERT_NUM_PROCS} --eval-interval -1 --save-interval -1")
    run_policy(run_settings)

    # Run BCO
    run_settings = GeneralRunSettings(BehavioralCloningFromObs, BasicPolicy, f"--prefix gail-test --env-name {TEST_ENV} --cuda False --traj-load-path {expected_trajs} --num-env-steps {EXPERT_NUM_ENV_SAMPLES} --num-steps {EXPERT_NUM_STEPS} --num-processes {EXPERT_NUM_PROCS} --eval-interval -1 --save-interval -1 --no-wb")
    run_policy(run_settings)


    os.remove(expected_file)
    os.remove(expected_trajs)


