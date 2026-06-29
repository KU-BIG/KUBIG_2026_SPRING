#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=tda-ms
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=0-47

# 8 데이터셋 × {full, baseline} × seed{0,1,2} = 48 run. mean±std 용. GPU 계속 포화.
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
unset PYTHONPATH || true
cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data

DATASETS=(acm dblp imdb freebase aminer mag dblp_pyg imdb_pyg)
MODES=(full baseline)
SEEDS=(0 1 2)
DS=${DATASETS[$((SLURM_ARRAY_TASK_ID / 6))]}
REM=$((SLURM_ARRAY_TASK_ID % 6))
MODE=${MODES[$((REM / 3))]}
SEED=${SEEDS[$((REM % 3))]}
TOPO=""; [ "$MODE" = "baseline" ] && TOPO="--no-topology"
OUT=runs/ms/${DS}_${MODE}_s${SEED}
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} date=$(date) ds=$DS mode=$MODE seed=$SEED"
python -m tda.train --config configs/${DS}.json --dataset ${DS} \
    --data-root "$DATA_ROOT" --output-dir "$OUT" --seed "$SEED" $TOPO
echo "done $OUT $(date)"
