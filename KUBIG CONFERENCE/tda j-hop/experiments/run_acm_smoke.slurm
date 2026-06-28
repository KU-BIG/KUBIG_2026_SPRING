#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=tda-acm-smoke
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# GPU 경로 빠른 검증 (full 파이프라인, 축소 config). 본 실험 전 sanity 용.
set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true
cd /mnt/data/users/junyoungpark/code/TDA
DATA_ROOT=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} date=$(date)"
python -m tda.train --config configs/acm_smoke.json --dataset acm \
    --data-root "$DATA_ROOT" --output-dir runs/acm_smoke
echo "done $(date)"
