#!/bin/bash
# 전체 A~D 실험을 모든 데이터셋에 대해 **한 명령**으로 실행 (slurm 하나, manifest 하나).
#   - config 자동 생성: 14 데이터셋 × 10 설정(A1/A3/B2/C2/D2/D3×2/D4×2/D5/topo-only) × seed{0..NSEED-1}
#   - **동시 실행 최대 8개 GPU** (%8 throttle + 청크간 drain 대기) — 절대 초과 안 함
#   - 이미 끝난 run 은 자동 skip(resume), 끝나면 results/SUMMARY.md(mean±std) 집계
#
# 사용:  bash experiments/run_campaign.sh            # seed 는 experiments/seeds.txt
#        MAXGPU=4 bash experiments/run_campaign.sh   # 동시 GPU 수 조절(기본 8)
set -euo pipefail
cd "$(dirname "$0")/.."
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn

echo "[1/3] config 생성 (14 데이터셋 × A~D)"
python experiments/gen_full_campaign.py
N=$(wc -l < configs/campaign/manifest.txt)
NSEED=$(grep -c . experiments/seeds.txt)           # 기록된 random seed 개수 (experiments/seeds.txt)
TOTAL=$((N * NSEED))
echo "  configs=$N, random seeds=$NSEED, tasks=$TOTAL"

echo "[2/3] 청크 제출 — **동시 실행 최대 ${MAXGPU:=8}개로 강제** (%${MAXGPU} throttle + 청크간 drain 대기)"
NEXT=0; CHUNK=90
while [ $NEXT -lt $TOTAL ]; do
    # 이전 청크가 완전히 끝날 때까지 대기 → 두 배열이 겹치지 않아 동시 ≤ MAXGPU 보장
    while [ "$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
    END=$((NEXT + CHUNK - 1)); [ $END -ge $TOTAL ] && END=$((TOTAL - 1))
    sbatch --array=${NEXT}-${END}%${MAXGPU} experiments/run_campaign.slurm \
        && echo "  submitted ${NEXT}-${END} (동시 ≤${MAXGPU})"
    NEXT=$((END + 1))
done

echo "[3/3] 완료 대기 후 집계"
while squeue -u "$USER" -h -r 2>/dev/null | grep -q tda-camp; do sleep 120; done
python experiments/regen_results.py
echo "done -> results/SUMMARY.md"
