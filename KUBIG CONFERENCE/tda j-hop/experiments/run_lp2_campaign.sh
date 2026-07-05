#!/bin/bash
# LP Level 2 캠페인: 3 ds × {lp2_base, lp2_real, lp2_noise, lp2_mix} × NSEED(기본 10), ≤MAXGPU.
#   - L2 전용 manifest/seeds 파일 + sbatch env 고정 → L1 과의 파일 race 원천 차단
#   - QOS 제출상한(100) 대응: 90개 청크 제출, 청크간 drain 대기
#   - warm(3 task): 데이터셋별 lp2_real 1개 → pair-EPD 캐시 생성 후 본대
# 실행:  MAXGPU=8 bash experiments/run_lp2_campaign.sh
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
MAXGPU=${MAXGPU:-4}
NSEED=${NSEED:-10}
MANIFEST=configs/campaign/lp2_manifest.txt
SEEDS_F=experiments/seeds_lp2.txt
EXP="--export=ALL,LP_MANIFEST=${MANIFEST},LP_SEEDS=${SEEDS_F}"

python experiments/gen_lp.py
for ds in acm dblp aifb; do
  for c in lp2_base lp2_real lp2_noise lp2_mix; do
    echo "configs/campaign/${ds}__${c}.json"
  done
done > "$MANIFEST"
head -${NSEED} experiments/seeds.txt > "$SEEDS_F"
NCFG=$(wc -l < "$MANIFEST")
TOTAL=$((NCFG * NSEED))
echo "configs=$NCFG seeds=$NSEED total=$TOTAL (동시 ≤$MAXGPU)"

# warm: 데이터셋별 lp2_real 첫 seed → pair-EPD 캐시 채움
WARM=""
for ds in acm dblp aifb; do
  i=$(grep -n "${ds}__lp2_real.json" "$MANIFEST" | head -1 | cut -d: -f1)
  WARM="$WARM,$(( (i-1) * NSEED ))"
done
WARM=${WARM#,}
wid=$(sbatch --parsable $EXP --array=${WARM}%3 experiments/run_lp.slurm)
echo "warm job $wid (tasks $WARM)"
# warm 완료 대기 (캐시 보장) 후 본대를 90개 청크로 (QOS 상한 회피)
sleep 20
while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
NEXT=0; CHUNK=90
while [ $NEXT -lt $TOTAL ]; do
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  END=$((NEXT + CHUNK - 1)); [ $END -ge $TOTAL ] && END=$((TOTAL - 1))
  sbatch $EXP --array=${NEXT}-${END}%${MAXGPU} experiments/run_lp.slurm \
    && echo "submitted ${NEXT}-${END} %${MAXGPU}"
  NEXT=$((END + 1))
done
echo "all chunks submitted"
