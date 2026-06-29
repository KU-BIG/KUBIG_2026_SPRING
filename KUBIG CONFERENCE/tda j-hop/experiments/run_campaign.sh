#!/bin/bash
# 전체 A~D 실험을 12개 데이터셋에 대해 **한 명령**으로 실행 (slurm 하나, manifest 하나).
#   - config 자동 생성: 12 데이터셋 × 10 설정(A1/A3/B2/C2/D2/D3×2/D4×2/D5/topo-only) × seed{0,1,2}
#   - QOS 상한(동시제출 100) 안에서 한 배열을 청크로 제출 → GPU 계속 포화
#   - 이미 끝난 run 은 자동 skip(resume), 끝나면 results/SUMMARY.md(mean±std) 집계
#
# 사용:  bash experiments/run_campaign.sh
# (D1=no-kNN 은 ego 폭증으로 비현실적이라 제외.)
set -euo pipefail
cd "$(dirname "$0")/.."
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn

echo "[1/3] config 생성 (12 데이터셋 × A~D)"
python experiments/gen_full_campaign.py
N=$(wc -l < configs/campaign/manifest.txt)
TOTAL=$((N * 3))                                  # × seed{0,1,2}
echo "  configs=$N, tasks=$TOTAL"

echo "[2/3] 단일 배열을 QOS 안에서 청크 제출 (GPU 포화)"
NEXT=0; CHUNK=80
while [ $NEXT -lt $TOTAL ]; do
    SUB=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l); ROOM=$((96 - SUB))
    if [ $ROOM -ge $CHUNK ]; then
        END=$((NEXT + CHUNK - 1)); [ $END -ge $TOTAL ] && END=$((TOTAL - 1))
        sbatch --array=${NEXT}-${END} experiments/run_campaign.slurm && echo "  submitted ${NEXT}-${END}"
        NEXT=$((END + 1))
    else
        sleep 90
    fi
done

echo "[3/3] 완료 대기 후 집계"
while squeue -u "$USER" -h -r 2>/dev/null | grep -q tda-camp; do sleep 120; done
python experiments/regen_results.py
echo "done -> results/SUMMARY.md"
