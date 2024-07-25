#!/bin/bash
# Specify the conda environment name to install to as the argument. 
git clone https://github.com/benelot/pybullet-gym.git envs/pybullet-gym
cd envs/pybullet-gym
~/miniconda3/envs/$1/bin/pip install -e .
