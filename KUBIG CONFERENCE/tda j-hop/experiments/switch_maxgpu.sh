#!/bin/bash
# 돌고 있는 캠페인의 동시 GPU 상한을 변경한다 (인자: 새 MAXGPU, 기본 4).
#   1) 현재 run_campaign.sh 컨트롤러 종료 (자기 자신/awk 제외)
#   2) 살아있는 tda-camp array job 의 ArrayTaskThrottle 을 NEW 로 (실행중 잡은 취소 안 함 →
#      작업 낭비 0, 12개는 끝나는 대로 ≤NEW 로 자연 감소, 새 잡은 NEW 초과 안 뜸)
#   3) 컨트롤러를 MAXGPU=NEW 로 재기동(detached) → 이후 청크도 %NEW 로 제출
set -uo pipefail
cd /mnt/data/users/junyoungpark/code/TDA
NEW=${1:-4}
LOG="slurm_logs/switch_$(date +%Y%m%d_%H%M%S).log"
{
  echo "[switch] $(date) target MAXGPU=$NEW"

  # 1) 기존 컨트롤러 종료
  for pid in $(ps -eo pid,cmd | awk '/bash experiments\/run_campaign\.sh/ && !/awk/ {print $1}'); do
    echo "[switch] kill controller pid=$pid"; kill "$pid" 2>/dev/null || true
  done
  sleep 3

  # 2) 살아있는 array job throttle 을 NEW 로 (취소 없이 즉시 상한만 낮춤)
  for jid in $(squeue -u "$USER" -h -r -n tda-camp -o "%F" 2>/dev/null | sort -u); do
    scontrol update jobid="$jid" arraytaskthrottle="$NEW" 2>&1 | grep -v "already finished" || true
    val=$(scontrol show job "$jid" 2>/dev/null | grep -o "ArrayTaskThrottle=[0-9]*" | head -1)
    echo "[switch] array $jid -> $val"
  done

  # 3) 컨트롤러 재기동 (run_campaign.sh 가 내부에서 conda activate + cd 함)
  nohup setsid bash -c "MAXGPU=$NEW bash experiments/run_campaign.sh" \
    > slurm_logs/campaign_controller.log 2>&1 &
  echo "[switch] controller relaunched MAXGPU=$NEW pid=$!"
  echo "[switch] done $(date)"
} >> "$LOG" 2>&1
echo "switch logged -> $LOG"
