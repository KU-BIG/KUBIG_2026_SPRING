#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Moore-AnimateAnyone: 전체 파이프라인 실행 스크립트
# 사용:
#   bash run.sh                     # 기본 실행
#   bash run.sh --frames 64         # 더 긴 영상
#   bash run.sh --low_vram          # VRAM 부족 시 (해상도/프레임 축소)
# ─────────────────────────────────────────────────────────────────────────────

set -e  # 오류 발생 시 즉시 종료

# ─── 파라미터 기본값 ──────────────────────────────────────────────────────────
WIDTH=512
HEIGHT=784
FRAMES=32
SEED=42
CFG=3.5
STEPS=30
REF_IMAGE="./inputs/ref.png"
DRIVING_VIDEO="./inputs/driving.mp4"

# ─── 인자 파싱 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --low_vram)
      WIDTH=384; HEIGHT=512; FRAMES=24
      shift ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --seed)   SEED="$2";   shift 2 ;;
    --cfg)    CFG="$2";    shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "======================================================"
echo "  Moore-AnimateAnyone  |  ${WIDTH}x${HEIGHT}  |  ${FRAMES} frames"
echo "======================================================"

# ─── Step 0: 환경 체크 ────────────────────────────────────────────────────────
echo ""
echo "[STEP 0] Environment check..."
python tools/check_env.py

# ─── Step 1: weights 다운로드 ─────────────────────────────────────────────────
echo ""
echo "[STEP 1] Download pretrained weights..."
python tools/download_weights.py

# ─── Step 2: 포즈 추출 ────────────────────────────────────────────────────────
POSE_VIDEO="./outputs/pose/$(basename ${DRIVING_VIDEO%.*})_kps.mp4"
echo ""
echo "[STEP 2] Extracting pose from driving video..."
echo "  Input : $DRIVING_VIDEO"
echo "  Output: $POSE_VIDEO"
python tools/vid2pose.py \
  --video_path "$DRIVING_VIDEO" \
  --output_path "$POSE_VIDEO"

# ─── Step 3: animation.yaml 업데이트 ──────────────────────────────────────────
echo ""
echo "[STEP 3] Updating animation.yaml with input paths..."
python - <<EOF
from omegaconf import OmegaConf
cfg = OmegaConf.load("./configs/prompts/animation.yaml")
cfg["test_cases"] = {"${REF_IMAGE}": ["${POSE_VIDEO}"]}
OmegaConf.save(cfg, "./configs/prompts/animation.yaml")
print("  animation.yaml updated.")
EOF

# ─── Step 4: inference 실행 ───────────────────────────────────────────────────
echo ""
echo "[STEP 4] Running animation inference..."
python scripts/pose2vid.py \
  --config ./configs/prompts/animation.yaml \
  -W $WIDTH \
  -H $HEIGHT \
  -L $FRAMES \
  --seed $SEED \
  --cfg $CFG \
  --steps $STEPS

echo ""
echo "======================================================"
echo "  Done. Check the output/ directory."
echo "======================================================"
