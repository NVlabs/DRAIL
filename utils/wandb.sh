# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#! /usr/bin/env bash

conf=${1}
result=$(python -m wandb sweep ${conf} 2>&1)
sweep=$(echo $result | sed "s/^.*wandb agent \([^[:space:]]*\).*$/\1/")
echo "wandb agent ${sweep}"
python -m wandb agent ${sweep}
