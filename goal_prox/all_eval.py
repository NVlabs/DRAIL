import sys
sys.path.insert(0, './')
from goal_prox.main import GoalProxSettings
from rlf.exp_mgr.auto_eval import full_auto_eval

full_auto_eval(lambda x: GoalProxSettings(x))
