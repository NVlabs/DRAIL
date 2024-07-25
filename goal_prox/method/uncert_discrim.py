from rlf.algos.il.gail import GailDiscrim
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.algos.il.base_irl import BaseIRLAlgo
from rlf.comps.ensemble import Ensemble
import rlf.rl.utils as rutils

class UncertGAIL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([UncertGailDiscrim(get_discrim), agent_updater], 1)

class UncertGailDiscrim(GailDiscrim):
    def _create_discrim(self):
        return Ensemble(super()._create_discrim, self.args.n_nets)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--n-nets', type=int, default=5)
        parser.add_argument('--uncert-scale', type=float, default=0.1)

    def _compute_discrim_reward(self, storage, step, add_info):
        reward = super()._compute_discrim_reward(storage, step, add_info)

        state = self._trans_agent_state(storage.get_obs(step))
        action = storage.actions[step]
        action = rutils.get_ac_repr(self.action_space, action)
        uncert = self._compute_disc_uncert(state, action)

        reward -= (uncert * self.args.uncert_scale)
        return reward

    def _compute_disc_uncert(self, state, action):
        return super()._compute_disc_val(state,action).std(0)

    def _compute_disc_val(self, state, action):
        return super()._compute_disc_val(state,action).mean(0)
