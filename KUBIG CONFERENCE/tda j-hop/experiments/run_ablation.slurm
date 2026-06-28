#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=tda-abl
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=0-32

# ACM ablation 캠페인: 11 설정 × seed 0/1/2 = 33 runs.
# 설정: A1/B2/C2(+A3,+permutation) + D1(no-kNN)/D2(no-MIN)/D3(ch2,ch8)/D4(L1,L3)/D5(random) + topology-only.
# 먼저 experiments/gen_ablation_configs.py 로 configs/ablation/*.json 생성 필요.
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true

cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data

mapfile -t CONFIGS < configs/ablation/manifest.txt
N=${#CONFIGS[@]}
CFG_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
CFG=${CONFIGS[$CFG_IDX]}
NAME=$(basename "$CFG" .json)
OUT=runs/abl/${NAME}_s${SEED}

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} date=$(date) cfg=$CFG seed=$SEED -> $OUT"
python -m tda.train --config "$CFG" --dataset acm \
    --data-root "$DATA_ROOT" --output-dir "$OUT" --seed "$SEED"
echo "done $OUT $(date)"
