#!/bin/bash
# 전체 A~D 실험을 모든 데이터셋에 대해 **한 번에** 실행한다 (따로따로 입력할 필요 없음).
#   - config 자동 생성: 12 데이터셋 × 10 설정(A1/A3/B2/C2/D2/D3×2/D4×2/D5/topo-only)
#   - SLURM 배열을 QOS 상한(동시제출 100) 안에서 청크로 제출 → GPU 계속 포화
#   - 끝나면 results/SUMMARY.md (mean±std) 자동 집계
#
# 사용:  bash experiments/run_campaign.sh
# (D1=no-kNN 은 ego 폭증으로 비현실적이라 제외. seed 0/1/2.)
set -euo pipefail
cd "$(dirname "$0")/.."
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn

echo "[1/4] config 생성"
python experiments/gen_full_campaign.py     # configs/campaign     (8 데이터셋)
python experiments/gen_campaign_rdf.py      # configs/campaign_rdf (RDF 4 데이터셋)

# QOS 상한 안에서 배열을 청크로 제출하는 헬퍼
submit_chunked () {   # $1=slurm 파일  $2=총 task 수
    local slurm=$1 total=$2 next=0 chunk=80 end sub room
    while [ $next -lt $total ]; do
        sub=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l); room=$((96 - sub))
        if [ $room -ge $chunk ]; then
            end=$((next + chunk - 1)); [ $end -ge $total ] && end=$((total - 1))
            sbatch --array=${next}-${end} "$slurm" && echo "  submitted ${slurm} ${next}-${end}"
            next=$((end + 1))
        else
            sleep 90
        fi
    done
}

echo "[2/4] 메인 8 데이터셋 × A~D 제출 (240 task)"
submit_chunked experiments/run_campaign.slurm 240
echo "[3/4] RDF 4 데이터셋 × A~D 제출 (120 task)"
submit_chunked experiments/run_campaign_rdf.slurm 120

echo "[4/4] 완료 대기 후 집계"
while squeue -u "$USER" -h -r 2>/dev/null | grep -qE "tda-camp|tda-crdf"; do sleep 120; done
python experiments/regen_results.py
echo "done -> results/SUMMARY.md"
