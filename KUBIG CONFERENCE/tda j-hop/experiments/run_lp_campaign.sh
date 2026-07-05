#!/bin/bash
# LP Level-1 캠페인: 13 config × NSEED(기본 5) seed, 동시 ≤MAXGPU(기본 4).
#   1단계 warm(3 task: 데이터셋별 lp_c 1개 → 위상 캐시 생성) → 2단계 전체(캐시 적중, 전부 빠름)
# 실행:  MAXGPU=7 NSEED=5 bash experiments/run_lp_campaign.sh
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=${MAXGPU:-4}
NSEED=${NSEED:-5}

python experiments/gen_lp.py
for ds in acm dblp aifb; do
  for c in lp_a lp_b1 lp_c lp_m; do echo "configs/campaign/${ds}__${c}.json"; done
done > configs/campaign/lp_manifest.txt
echo "configs/campaign/dblp__lp_cx.json" >> configs/campaign/lp_manifest.txt
head -${NSEED} experiments/seeds.txt > experiments/seeds_lp.txt
NCFG=$(wc -l < configs/campaign/lp_manifest.txt)
TOTAL=$((NCFG * NSEED))
echo "configs=$NCFG seeds=$NSEED total=$TOTAL (동시 ≤$MAXGPU)"

# 1) warm: 데이터셋별 lp_c 1개(첫 seed)로 위상 캐시 채움 — 이후 모든 run 은 캐시 적중
WARM=""
for ds in acm dblp aifb; do
  i=$(grep -n "${ds}__lp_c.json" configs/campaign/lp_manifest.txt | head -1 | cut -d: -f1)
  WARM="$WARM,$(( (i-1) * NSEED ))"
done
WARM=${WARM#,}
wid=$(sbatch --parsable --array=${WARM}%3 experiments/run_lp.slurm)
echo "warm job $wid (tasks $WARM)"
# 2) 전체 (warm 뒤에)
sbatch --dependency=afterany:${wid} --array=0-$((TOTAL-1))%${MAXGPU} experiments/run_lp.slurm
echo "submitted full array (resume skip 포함)"
