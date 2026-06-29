#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=tda-mabl
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=0-39

# 8 데이터셋 × {B2,D2,D3,D4,D5} ablation = 40 runs. seed 0.
# 먼저: python experiments/gen_multidomain_ablation.py
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true
cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data

mapfile -t CFGS < configs/abl_multi/manifest.txt
CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}
NAME=$(basename "$CFG" .json)        # e.g. dblp__d5_random
DS=$(python -c "import json;print(json.load(open('$CFG'))['dataset'])")
OUT=runs/abl_multi/${NAME}
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} date=$(date) cfg=$CFG ds=$DS"
python -m tda.train --config "$CFG" --dataset "$DS" \
    --data-root "$DATA_ROOT" --output-dir "$OUT" --seed 0
echo "done $OUT $(date)"
