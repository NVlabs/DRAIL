# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

pip install -U gdown
current_directory=$(dirname "$(realpath "$0")")
expert_datasets_path="$current_directory/../expert_datasets"
python ${current_directory}/download_demos.py --dir $expert_datasets_path

python ${current_directory}/clip_push.py
python ${current_directory}/clip_pick.py
# python ${current_directory}/clip_walker.py
# python ${current_directory}/../shape_env/rollout_sine.py
