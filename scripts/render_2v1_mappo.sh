#!/bin/sh

env="MAPPOTraining2v1"
scenario="2v1/NoWeapon/MAPPOTraining"
algo="mappo"
exp="v1_2v1_mappo_render"
seed=0

# Path to the trained MAPPO model
MODEL_DIR="scripts/results/MAPPOTraining2v1/2v1/NoWeapon/MAPPOTraining/mappo/v1_2v1_mappo_training/wandb/latest-run/files"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo "Loading MAPPO model from: ${MODEL_DIR}"

# Set wandb to offline mode
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=0 python renders/render_2v1_mappo.py \
    --model-dir ${MODEL_DIR} \
    --output ${exp}.txt.acmi \
    --scenario ${scenario} \
    --use-latest