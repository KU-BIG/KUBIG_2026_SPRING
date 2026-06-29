#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=tda-multi
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=0-7

# 여러 도메인 이종그래프 NC, 8 GPU 병렬:
#   학술(ACM, DBLP) · 영화(IMDB) · 지식그래프(Freebase) × {full(C2), baseline(A1)}.
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true
cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data

DATASETS=(acm dblp imdb freebase acm dblp imdb freebase)
MODES=(full full full full baseline baseline baseline baseline)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
OUT=runs/multi/${DS}_${MODE}
TOPO=""; [ "$MODE" = "baseline" ] && TOPO="--no-topology"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} date=$(date) ds=$DS mode=$MODE"
python -m tda.train --config configs/${DS}.json --dataset ${DS} \
    --data-root "$DATA_ROOT" --output-dir "$OUT" --seed 0 $TOPO
echo "done $OUT $(date)"
