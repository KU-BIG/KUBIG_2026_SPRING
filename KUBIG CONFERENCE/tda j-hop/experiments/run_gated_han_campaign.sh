#!/bin/bash
# 실험 ①-HAN: 3 ds × 7조건(gh_*) × 10 seed = 210 task, ≤MAXGPU. 전용 manifest(env 고정).
# 위상 캐시는 gt 캠페인이 이미 생성(동일 키) → warm 불필요, 전 task 캐시 적중.
# 실행:  MAXGPU=8 bash experiments/run_gated_han_campaign.sh
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=${MAXGPU:-4}
MANIFEST=configs/campaign/gated_han_manifest.txt
EXP="--export=ALL,GATED_MANIFEST=${MANIFEST}"

python experiments/gen_gated.py
for ds in acm dblp aifb; do
  for c in gh_base gh_cat_real gh_cat_noise gh_cat_mix gh_gate_real gh_gate_noise gh_gate_mix; do
    echo "configs/campaign/${ds}__${c}.json"
  done
done > "$MANIFEST"
NSEED=$(grep -c . experiments/seeds.txt)
NCFG=$(wc -l < "$MANIFEST")
TOTAL=$((NCFG * NSEED))
echo "configs=$NCFG seeds=$NSEED total=$TOTAL (동시 ≤$MAXGPU)"

NEXT=0; CHUNK=90
while [ $NEXT -lt $TOTAL ]; do
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  END=$((NEXT + CHUNK - 1)); [ $END -ge $TOTAL ] && END=$((TOTAL - 1))
  sbatch $EXP --array=${NEXT}-${END}%${MAXGPU} experiments/run_gated.slurm \
    && echo "submitted ${NEXT}-${END} %${MAXGPU}" || echo "SBATCH FAILED ${NEXT}-${END}"
  NEXT=$((END + 1))
done
echo "all chunks submitted"
