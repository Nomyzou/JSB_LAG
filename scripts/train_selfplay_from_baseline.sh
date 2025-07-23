#!/bin/sh
env="SingleCombat"
scenario="1v1/NoWeapon/Selfplay"
algo="ppo"
exp="v1_selfplay_from_baseline"
seed=1

# --- IMPORTANT ---
# This path points to the latest wandb run directory for the vsBaseline experiment.
# It should contain the 'actor_latest.pt' and 'critic_latest.pt' files.
MODEL_DIR="scripts/results/SingleCombat/1v1/NoWeapon/vsBaseline/ppo/v1/wandb/latest-run" 

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo "Loading initial model from: ${MODEL_DIR}"

CUDA_VISIBLE_DEVICES=1 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 10 \
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 \
    --model-dir ${MODEL_DIR} \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --user-name "jyh" --wandb-name "thu_jsbsim_selfplay" 