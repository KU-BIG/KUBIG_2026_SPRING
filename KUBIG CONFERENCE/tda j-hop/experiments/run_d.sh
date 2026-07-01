#!/bin/bash
# (d) RGCN-only: 7개 데이터셋 × seeds, 동시 GPU ≤7.
# yelp(이전 array) drain → 결과 push → (d) 제출(%7) → (d) drain → 결과 push.
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=7
LOG="slurm_logs/run_d_$(date +%Y%m%d_%H%M%S).log"

push_results() {  # regen 새 결과표 + 커밋/푸시 (실패해도 계속)
  python experiments/regen_new.py >/dev/null 2>&1 || true
  git add results/SUMMARY.md 2>/dev/null || true
  git commit -q -m "Update results/SUMMARY: $1

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
  git pull --rebase -q origin main 2>/dev/null || true
  git push -q origin main 2>/dev/null && echo "[run_d] pushed results ($1)" || echo "[run_d] push skipped/failed ($1)"
}

{
  echo "[run_d] $(date) wait for current jobs (yelp) to drain (cap<=$MAXGPU)"
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  echo "[run_d] $(date) yelp done -> push (a)(c) results"
  push_results "after yelp (a)(c) complete"
  echo "[run_d] $(date) submit (d) RGCN-only %$MAXGPU"
  for ds in acm dblp imdb freebase mag aifb yelp; do
    echo "configs/campaign/${ds}__d_rgcn.json"
  done > configs/campaign/manifest.txt
  NSEED=$(grep -c . experiments/seeds.txt); N=$((7 * NSEED))
  sbatch --array=0-$((N - 1))%${MAXGPU} experiments/run_campaign.slurm \
    && echo "[run_d] submitted (d) array 0-$((N-1)) %${MAXGPU} (tasks=$N)"
  sleep 30
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  echo "[run_d] $(date) (d) done -> push (a)(c)(d) results"
  push_results "after (d) RGCN-only complete"
  echo "[run_d] all done $(date)"
} >> "$LOG" 2>&1
echo "run_d logged -> $LOG"
