import os
from typing import Dict

import attr
import numpy as np
import pandas as pd

import rlf.rl.utils as rutils


@attr.s(auto_attribs=True, slots=True)
class RunResult:
    prefix: str
    eval_result: Dict = {}

def plot_map(run_settings, runner=None):
    if runner is None:
        runner = run_settings.create_runner()
    end_update = runner.updater.get_num_updates()
    args = runner.args
    print("runner.should_load_from_checkpoint():", runner.should_load_from_checkpoint())
    if runner.should_load_from_checkpoint():
        runner.load_from_checkpoint()
    #import ipdb; ipdb.set_trace()
    sth = runner.plot_map(run_settings.create_traj_saver, dimension=2)

def plot_map_3d(run_settings, runner=None):
    if runner is None:
        runner = run_settings.create_runner()
    end_update = runner.updater.get_num_updates()
    args = runner.args
    print("runner.should_load_from_checkpoint():", runner.should_load_from_checkpoint())
    if runner.should_load_from_checkpoint():
        runner.load_from_checkpoint()
    #import ipdb; ipdb.set_trace()
    sth = runner.plot_map(run_settings.create_traj_saver, dimension=3)

def plot_density_subtraction(run_settings, runner=None):
    if runner is None:
        runner = run_settings.create_runner()
    end_update = runner.updater.get_num_updates()
    args = runner.args
    print("runner.should_load_from_checkpoint():", runner.should_load_from_checkpoint())
    if runner.should_load_from_checkpoint():
        runner.load_from_checkpoint()
    #import ipdb; ipdb.set_trace()
    sth = runner.plot_density(run_settings.create_traj_saver, operation='substraction')

def plot_density_division(run_settings, runner=None):
    if runner is None:
        runner = run_settings.create_runner()
    end_update = runner.updater.get_num_updates()
    args = runner.args
    print("runner.should_load_from_checkpoint():", runner.should_load_from_checkpoint())
    if runner.should_load_from_checkpoint():
        runner.load_from_checkpoint()
    #import ipdb; ipdb.set_trace()
    sth = runner.plot_density(run_settings.create_traj_saver, operation='division')