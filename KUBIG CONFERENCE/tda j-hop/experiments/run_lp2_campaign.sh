#!/bin/bash
# LP Level 2 캠페인: 3 ds × {lp2_base, lp2_real, lp2_noise, lp2_mix} × NSEED(기본 10), ≤MAXGPU.
#   warm(3 task: 데이터셋별 lp2_real 1개 → pair-EPD 캐시 생성) → 본대(캐시 적중).
# 실행:  MAXGPU=7 bash experiments/run_lp2_campaign.sh
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=${MAXGPU:-4}
NSEED=${NSEED:-10}

python experiments/gen_lp.py
for ds in acm dblp aifb; do
  for c in lp2_base lp2_real lp2_noise lp2_mix; do
    echo "configs/campaign/${ds}__${c}.json"
  done
done > configs/campaign/lp_manifest.txt
head -${NSEED} experiments/seeds.txt > experiments/seeds_lp.txt
NCFG=$(wc -l < configs/campaign/lp_manifest.txt)
TOTAL=$((NCFG * NSEED))
echo "configs=$NCFG seeds=$NSEED total=$TOTAL (동시 ≤$MAXGPU)"

WARM=""
for ds in acm dblp aifb; do
  i=$(grep -n "${ds}__lp2_real.json" configs/campaign/lp_manifest.txt | head -1 | cut -d: -f1)
  WARM="$WARM,$(( (i-1) * NSEED ))"
done
WARM=${WARM#,}
wid=$(sbatch --parsable --array=${WARM}%3 experiments/run_lp.slurm)
echo "warm job $wid (tasks $WARM)"
sbatch --dependency=afterany:${wid} --array=0-$((TOTAL-1))%${MAXGPU} experiments/run_lp.slurm
echo "submitted full array"
