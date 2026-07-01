#!/bin/bash
# B1/D1 noisy-topology(random): 7 ds × {b1_noise(HAN), d1_noise(RGCN)} × seeds, 동시 GPU ≤4.
# 밤에 실행:  nohup setsid bash experiments/run_noise.sh > slurm_logs/run_noise_launcher.log 2>&1 &
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=4
LOG="slurm_logs/run_noise_$(date +%Y%m%d_%H%M%S).log"

push_results() {
  python experiments/regen_new.py >/dev/null 2>&1 || true
  git add results/SUMMARY.md 2>/dev/null || true
  git commit -q -m "results

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
  git pull --rebase -q origin main 2>/dev/null || true
  git push -q origin main 2>/dev/null && echo "[run_noise] pushed results" || echo "[run_noise] push skipped"
}

{
  echo "[run_noise] $(date) wait for queue to drain (cap<=$MAXGPU)"
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  for ds in acm dblp imdb freebase mag aifb yelp; do
    echo "configs/campaign/${ds}__b1_noise.json"
    echo "configs/campaign/${ds}__d1_noise.json"
  done > configs/campaign/manifest.txt
  NSEED=$(grep -c . experiments/seeds.txt); N=$((14 * NSEED))
  sbatch --array=0-$((N - 1))%${MAXGPU} experiments/run_campaign.slurm \
    && echo "[run_noise] submitted 0-$((N-1)) %${MAXGPU} (tasks=$N)"
  sleep 30
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  push_results
  echo "[run_noise] done $(date)"
} >> "$LOG" 2>&1
echo "run_noise logged -> $LOG"
