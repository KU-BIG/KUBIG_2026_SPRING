#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=tda-gated
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# 실험 ①: manifest(configs/campaign/gated_manifest.txt) × seeds 배열.
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true
cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data

mapfile -t CFGS < configs/campaign/gated_manifest.txt
mapfile -t SEEDS < experiments/seeds.txt
NSEED=${#SEEDS[@]}
CFG=${CFGS[$((SLURM_ARRAY_TASK_ID / NSEED))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % NSEED))]}
NAME=$(basename "$CFG" .json)
DS=$(python -c "import json;print(json.load(open('$CFG'))['dataset'])")
OUT=runs/gated/${NAME}_s${SEED}
if [ -f "$OUT/metrics.json" ]; then echo "skip: $OUT"; exit 0; fi
echo "host=$(hostname) date=$(date) cfg=$CFG seed=$SEED"
python -m tda.gated --config "$CFG" --dataset "$DS" --data-root "$DATA_ROOT" \
    --output-dir "$OUT" --seed "$SEED"
echo "done $OUT $(date)"
