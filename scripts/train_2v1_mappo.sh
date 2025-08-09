#!/bin/sh

env="MAPPOTraining2v1"
scenario="2v1/NoWeapon/MAPPOTraining"
algo="mappo"
exp="v1_2v1_mappo_training"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

# Set wandb to offline mode to avoid login issues
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=0 python scripts/train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 4 --cuda \
    --num-env-steps 10000000 \
    --episode-length 1000 \
    --num-mini-batch 1 \
    --ppo-epoch 10 \
    --seed 0 \
    --cuda \
    --log-interval 1 \
    --save-interval 100 \
    --use-wandb False \
    --use-max-grad-norm \
    --max-grad-norm 0.5 \
    --clip-param 0.2 \
    --use-clipped-value-loss \
    --value-loss-coef 0.5 \
    --entropy-coef 0.01 \
    --lr 3e-4 \
    --gae-lambda 0.95 \
    --gamma 0.99