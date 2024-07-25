from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO

class ProxIL(NestedAlgo):
    def __init__(self, agent_updater=PPO()):
        super().__init__([ProxFunc(), agent_updater], designated_rl_idx=1)



