#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=tda-acm
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=0-5

# ACM 실험: baseline(HAN 단독) vs full(GTN+PDGNN+HAN, attention fusion), seed 0/1/2.
# 인덱스 0-2 = baseline, 3-5 = full.  (HAN only / attention fusion only — 사용자 지정 범위)
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true

cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data

MODES=(baseline baseline baseline full full full)
SEEDS=(0 1 2 0 1 2)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=runs/acm_${MODE}_s${SEED}

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} date=$(date) mode=$MODE seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

TOPO_FLAG=""
if [ "$MODE" = "baseline" ]; then TOPO_FLAG="--no-topology"; fi

python -m tda.train --config configs/acm.json --dataset acm \
    --data-root "$DATA_ROOT" --output-dir "$OUT" --seed "$SEED" $TOPO_FLAG

echo "done $OUT $(date)"
