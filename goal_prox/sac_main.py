"""
Code is heavily based off of https://github.com/denisyarats/pytorch_sac.
The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
"""
import sys

sys.path.insert(0, "./")

from functools import partial

import d4rl
import torch.nn as nn
from rlf import run_policy
from rlf.algos import (GAIL, PPO, BaseAlgo, BehavioralCloning,
                       BehavioralCloningFromObs, BehavioralCloningPretrain,
                       GailDiscrim)
from rlf.algos.il.base_il import BaseILAlgo
from rlf.algos.il.gaifo import GAIFO
from rlf.algos.il.sqil import SQIL
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.off_policy.sac import SAC
from rlf.args import str2bool
from rlf.policies import BasicPolicy, DistActorCritic, RandomPolicy
from rlf.policies.action_replay_policy import ActionReplayPolicy
from rlf.policies.actor_critic.dist_actor_q import (DistActorQ, get_sac_actor,
                                                    get_sac_critic)
from rlf.policies.actor_critic.reg_actor_critic import RegActorCritic
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.rl.loggers.wb_logger import (WbLogger, get_wb_ray_config,
                                      get_wb_ray_kwargs)
from rlf.rl.model import CNNBase, MLPBase, MLPBasic, TwoLayerMlpWithAction
from rlf.run_settings import RunSettings

from examples.test_run_settings import TestRunSettings
#from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic
import torch.nn.functional as F
from rlf.rl.model import BaseNet, IdentityBase, MLPBase
import torch
import math
from functools import partial


class SACRunSettings(TestRunSettings):
    def get_policy(self):
        return DistActorQ(
				get_critic_fn=partial(get_sac_critic, hidden_dim=self.base_args.hidden_dim),
                get_actor_fn=partial(get_sac_actor, hidden_dim=self.base_args.hidden_dim)
                )

    def get_algo(self):
        return SAC()

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--hidden-dim', type=int, default=1024)

if __name__ == "__main__":
    run_policy(SACRunSettings())
