#!/bin/bash
# 실험 ① 캠페인: 3 ds × 7조건 × 10 seed = 210 task, 동시 ≤MAXGPU(기본 4).
#   warm(3 task: 데이터셋별 gt_gate_real 1개 → topo_seed 고정 위상 캐시 생성) → 본대(캐시 적중).
# 실행:  MAXGPU=7 bash experiments/run_gated_campaign.sh
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=${MAXGPU:-4}

python experiments/gen_gated.py
for ds in acm dblp aifb; do
  for c in gt_base gt_cat_real gt_cat_noise gt_cat_mix gt_gate_real gt_gate_noise gt_gate_mix; do
    echo "configs/campaign/${ds}__${c}.json"
  done
done > configs/campaign/gated_manifest.txt
NSEED=$(grep -c . experiments/seeds.txt)
NCFG=$(wc -l < configs/campaign/gated_manifest.txt)
TOTAL=$((NCFG * NSEED))
echo "configs=$NCFG seeds=$NSEED total=$TOTAL (동시 ≤$MAXGPU)"

# warm: 데이터셋별 gt_gate_real 첫 seed → 위상 캐시 채움 (본대는 전부 캐시 적중)
WARM=""
for ds in acm dblp aifb; do
  i=$(grep -n "${ds}__gt_gate_real.json" configs/campaign/gated_manifest.txt | head -1 | cut -d: -f1)
  WARM="$WARM,$(( (i-1) * NSEED ))"
done
WARM=${WARM#,}
wid=$(sbatch --parsable --array=${WARM}%3 experiments/run_gated.slurm)
echo "warm job $wid (tasks $WARM)"
# 본대는 QOS 제출상한(100) 때문에 90개 청크로 나눠 제출 (warm 뒤, 청크간 drain)
NEXT=0; CHUNK=90; DEP="--dependency=afterany:${wid}"
while [ $NEXT -lt $TOTAL ]; do
  if [ $NEXT -gt 0 ]; then
    while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
    DEP=""
  fi
  END=$((NEXT + CHUNK - 1)); [ $END -ge $TOTAL ] && END=$((TOTAL - 1))
  sbatch $DEP --array=${NEXT}-${END}%${MAXGPU} experiments/run_gated.slurm \
    && echo "submitted ${NEXT}-${END} %${MAXGPU}"
  NEXT=$((END + 1))
done
echo "all submitted"
