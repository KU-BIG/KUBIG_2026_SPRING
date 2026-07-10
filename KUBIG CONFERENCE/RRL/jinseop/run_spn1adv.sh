#!/bin/bash
cd /workspace/multiHRI

echo "=== SPN_1ADV 학습 시작 ==="
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PYTHONPATH=/workspace/multiHRI python scripts/train_agents.py \
    --layout-names "3_chefs_counter_circuit" \
    --algo-name SPN_1ADV \
    --num-players 3 \
    --total-ego-agents 2 \
    --teammates-len 2 \
    --max-concurrent-jobs 3 \
    --n-x-sp-total-training-timesteps 15000000 \
    --adversary-total-training-timesteps 15000000 \
    --pop-total-training-timesteps 30000000 \
    --epoch-timesteps 50000 \
    --eval-steps-interval 20 \
    --n-envs 384 \
    --batch-size 128 \
    --num-of-ckpoints 40 \
    --custom-agent-ck-rate-generation 4 \
    --num-steps-in-traj-for-dyn-adv 2 \
    --num-static-advs-per-heatmap 1 \
    --num-dynamic-advs-per-heatmap 1 \
    --use-val-func-for-heatmap-gen false \
    --prioritized-sampling false \
    --pop-force-training false \
    --adversary-force-training false \
    --primary-force-training false \
    --how-long 20 \
    --exp-dir "Classic/3" \
    --wandb-mode online \
    2>&1 | tee /workspace/multiHRI/logs/train_spn1adv.log
