#!/bin/bash
# B1/D1 noisy-topology(random): 7 ds × {b1_noise(HAN), d1_noise(RGCN)} × seeds.
# 동시 GPU ≤7, QOS 제출상한(100) 회피 위해 90개씩 청크 제출(청크간 drain 대기). 끝나면 결과 push.
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=7
CHUNK=90
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
  for ds in acm dblp imdb freebase mag aifb yelp; do
    echo "configs/campaign/${ds}__b1_noise.json"
    echo "configs/campaign/${ds}__d1_noise.json"
  done > configs/campaign/manifest.txt
  NSEED=$(grep -c . experiments/seeds.txt); TOTAL=$((14 * NSEED))
  echo "[run_noise] $(date) TOTAL=$TOTAL tasks, chunk=$CHUNK, %$MAXGPU"
  NEXT=0
  while [ $NEXT -lt $TOTAL ]; do
    while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 30; done
    END=$((NEXT + CHUNK - 1)); [ $END -ge $TOTAL ] && END=$((TOTAL - 1))
    sbatch --array=${NEXT}-${END}%${MAXGPU} experiments/run_campaign.slurm \
      && echo "[run_noise] submitted ${NEXT}-${END} %${MAXGPU}"
    NEXT=$((END + 1))
  done
  sleep 30
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 30; done
  push_results
  echo "[run_noise] done $(date)"
} >> "$LOG" 2>&1
echo "run_noise logged -> $LOG"
