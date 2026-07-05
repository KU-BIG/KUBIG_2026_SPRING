#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=tda-lp
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# LP Level-1: manifest(configs/campaign/lp_manifest.txt) × seeds(LP_SEEDS 파일) 배열.
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true
cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data

# 캠페인별 파일을 env 로 고정 (공유 파일 race 방지 — L1/L2 가 서로 덮어쓰던 사고 재발 금지)
mapfile -t CFGS < "${LP_MANIFEST:-configs/campaign/lp_manifest.txt}"
mapfile -t SEEDS < "${LP_SEEDS:-experiments/seeds_lp.txt}"
NSEED=${#SEEDS[@]}
CFG=${CFGS[$((SLURM_ARRAY_TASK_ID / NSEED))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % NSEED))]}
NAME=$(basename "$CFG" .json)
DS=$(python -c "import json;print(json.load(open('$CFG'))['dataset'])")
OUT=runs/lp/${NAME}_s${SEED}
if [ -f "$OUT/metrics.json" ]; then echo "skip: $OUT"; exit 0; fi
echo "host=$(hostname) date=$(date) cfg=$CFG seed=$SEED"
python -m tda.lp --config "$CFG" --dataset "$DS" --data-root "$DATA_ROOT" \
    --output-dir "$OUT" --seed "$SEED"
echo "done $OUT $(date)"
