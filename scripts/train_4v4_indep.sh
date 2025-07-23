#!/bin/sh

env="IndepCombatEnv"
scenario="4v4/NoWeapon/IndepHierarchySelfplay"
algo="ppo"
exp="v1_4v4_indep"
seed=1

# --- IMPORTANT ---
# This path should point to your best 1v1 SELF-PLAY model directory.
# This model will be used as the initial policy for all 4 agents.
MODEL_DIR="scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/v1_selfplay_from_baseline/run1/models"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo "Loading initial model for all agents from: ${MODEL_DIR}"

CUDA_VISIBLE_DEVICES=1 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 8 --cuda \
    --log-interval 1 --save-interval 10 \
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 \
    --model-dir ${MODEL_DIR} \
    --num-mini-batch 5 --buffer-size 5000 --num-env-steps 5e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "256 256" --act-hidden-size "256 256" --recurrent-hidden-size 256 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --user-name "your_name" --wandb-name "jsbsim_4v4_indep" 