#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=tda-camp
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=0-239

# 통합 캠페인: 8 데이터셋 × 10 설정(A~D) × seed{0,1,2} = 240 task.
# 240개를 큐에 쌓아 가용 GPU 를 계속 포화. 먼저: python experiments/gen_full_campaign.py
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
unset PYTHONPATH || true
cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data

mapfile -t CFGS < configs/campaign/manifest.txt
N=${#CFGS[@]}
CFG=${CFGS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
NAME=$(basename "$CFG" .json)          # e.g. dblp__d5_random
DS=$(python -c "import json;print(json.load(open('$CFG'))['dataset'])")
OUT=runs/campaign/${NAME}_s${SEED}
if [ -f "$OUT/metrics.json" ]; then echo "skip (이미 완료): $OUT"; exit 0; fi   # resume
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} date=$(date) cfg=$CFG ds=$DS seed=$SEED"
python -m tda.train --config "$CFG" --dataset "$DS" \
    --data-root "$DATA_ROOT" --output-dir "$OUT" --seed "$SEED"
echo "done $OUT $(date)"
