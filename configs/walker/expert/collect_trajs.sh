# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

python drail/main.py \
    --load-file ./expert_datasets/ppo_walker_expert_model.pt \
    --alg=ppo \
    --clip-actions=True \
    --cuda=True \
    --entropy-coef=0.001 \
    --env-name=Walker2d-v3 \
    --eval-interval=20000 \
    --eval-num-processes=1 \
    --log-interval=1 \
    --lr=0.0001 \
    --max-grad-norm=0.5 \
    --normalize-env=True \
    --num-env-steps=25000000 \
    --num-epochs=10 \
    --num-eval=25 \
    --num-mini-batch=32 \
    --num-render=1000 \
    --num-steps=256 \
    --ppo-hidden-dim=256 \
    --prefix=ppo \
    --save-interval=100000 \
    --seed=1 \
    --vid-fps=100 \
    --eval-only \
    --eval-save \
    --num-processes 1