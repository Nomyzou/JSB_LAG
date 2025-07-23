#!/bin/sh

env="IndepCombatEnv"
scenario="4v4/NoWeapon/IndepFixedPairing"
# This is an evaluation, not training, so algo and exp are just for logging.
algo="ppo"
exp="v1_4v4_fixed_pairing_eval"
seed=1

# --- IMPORTANT ---
# This path should point to your best 1v1 SELF-PLAY model directory.
# This model will be used by all 8 agents (both Red and Blue).
MODEL_DIR="scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/v1_selfplay_from_baseline/run1/models"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo "Loading 1v1 model for all agents from: ${MODEL_DIR}"

# Note: We are using train_jsbsim.py, but it's only for running the environment.
# No actual training will occur because the Task doesn't use the RL action.
# We set n_rollout_threads to 1 for clear, non-parallel evaluation.
CUDA_VISIBLE_DEVICES=1 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 1 --cuda \
    --model-dir ${MODEL_DIR} \
    --use-wandb 