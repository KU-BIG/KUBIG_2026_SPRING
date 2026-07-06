#!/bin/bash
# 실험 ①′ — 고정 GTN+PDGNN 위상 factorial (5 ds × {gt2,gh2} × {cat,gate}×{real,mix} × 10 seed = 400)
# base/noise 조건은 위상 무관이라 기존 runs/gated 재사용. 위상 = GTN(topo_seed 고정, 1회 학습) 채널.
# warm: 데이터셋당 1 task 로 npz 캐시 생성 → drain → 본대 청크(≤90) %MAXGPU.
# 실행:  MAXGPU=4 bash experiments/run_gated2_campaign.sh
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
MAXGPU=${MAXGPU:-4}
GM=configs/campaign/gated2_manifest.txt
N=$(( $(wc -l < "$GM") * 10 ))
echo "total tasks=$N (동시 ≤$MAXGPU)"

# ---- warm: 각 데이터셋 첫 config(gt2_cat_real) × 첫 seed = task {0,80,160,240,320} ----
w=$(sbatch --parsable --export=ALL,GATED_MANIFEST=$GM --array=0,80,160,240,320%4 experiments/run_gated.slurm) \
  && echo "warm job $w" || { echo "SBATCH FAILED warm"; exit 1; }
sleep 30
while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
NC_CACHE=$(ls cache/topo_gtnfix/*.npz 2>/dev/null | wc -l)
if [ "$NC_CACHE" -lt 5 ]; then
  echo "ABORT: warm 후 캐시 $NC_CACHE/5 — 본대 미제출 (각 run 이 위상 재계산하는 폭주 방지)"
  exit 1
fi
echo "warm done (캐시 $NC_CACHE/5) -> mains"

# ---- 본대: 400 task, 청크 ≤90 (QOS 100 제한), 청크 간 drain ----
NEXT=0; CHUNK=90
while [ $NEXT -lt $N ]; do
  while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
  END=$((NEXT + CHUNK - 1)); [ $END -ge $N ] && END=$(( N - 1 ))
  sbatch --export=ALL,GATED_MANIFEST=$GM --array=${NEXT}-${END}%${MAXGPU} experiments/run_gated.slurm \
    && echo "submitted ${NEXT}-${END} %${MAXGPU}" || echo "SBATCH FAILED ${NEXT}-${END}"
  NEXT=$((END + 1))
done
echo "all submitted"
