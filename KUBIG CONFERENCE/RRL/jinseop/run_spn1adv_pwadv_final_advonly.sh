#!/bin/bash
cd /workspace
source /workspace/venv/bin/activate

echo "=== SPN_1ADV PWADV_final (adv-only, reusing round2 ADV_2) 학습 시작 ==="
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PYTHONPATH=/workspace python scripts/train_agents.py \
    --layout-names "3_chefs_counter_circuit" \
    --algo-name SPN_1ADV_PWADV_FINAL_ADVONLY \
    --num-players 3 \
    --teammates-len 2 \
    --n-x-sp-total-training-timesteps 15000000 \
    --epoch-timesteps 50000 \
    --eval-steps-interval 20 \
    --n-envs 384 \
    --batch-size 128 \
    --num-of-ckpoints 40 \
    --primary-force-training false \
    --exp-dir "Classic/3" \
    --wandb-mode online \
    2>&1 | tee /workspace/logs/train_spn1adv_pwadv_final_advonly.log
