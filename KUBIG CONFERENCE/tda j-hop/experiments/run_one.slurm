#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=tda-one
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=0-1

# 한 데이터셋 full(C2) + baseline(A1). DS 환경변수로 지정:
#   sbatch --export=ALL,DS=imdb experiments/run_one.slurm
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true
cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data
: "${DS:?set DS=<dataset>}"

MODES=(full baseline)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
TOPO=""; [ "$MODE" = "baseline" ] && TOPO="--no-topology"
OUT=runs/multi/${DS}_${MODE}
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} date=$(date) ds=$DS mode=$MODE"
python -m tda.train --config configs/${DS}.json --dataset ${DS} \
    --data-root "$DATA_ROOT" --output-dir "$OUT" --seed 0 $TOPO
echo "done $OUT $(date)"
