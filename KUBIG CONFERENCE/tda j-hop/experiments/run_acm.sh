#!/bin/bash
# ACM 실험 실행. ablation 을 켜고 끌 수 있다.
#   bash experiments/run_acm.sh off   # 메인만 (A1 baseline + C2 full, seed 0/1/2)
#   bash experiments/run_acm.sh on    # 메인 + 전체 ablation (D1~D6)
set -euo pipefail
ABLATION=${1:-off}
cd "$(dirname "$0")/.."

echo "[run] 메인 실험 제출 (A1 baseline + C2 full)"
sbatch experiments/run_acm.slurm

if [ "$ABLATION" = "on" ]; then
    echo "[run] ablation ON -> 설정 생성 + D1~D6 제출"
    python experiments/gen_ablation_configs.py
    sbatch experiments/run_ablation.slurm
else
    echo "[run] ablation OFF (켜려면: bash experiments/run_acm.sh on)"
fi
