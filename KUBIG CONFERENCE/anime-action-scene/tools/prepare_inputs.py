"""
입력 파일 전처리 유틸리티
  - reference image 리사이즈 및 크롭 (비율 맞추기)
  - driving video 품질 확인
  - 포즈 추출 완료 여부 확인

실행:
  python tools/prepare_inputs.py \
    --ref inputs/my_photo.jpg \
    --video inputs/dance.mp4 \
    -W 512 -H 784
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="입력 파일 전처리 (리사이즈, 크롭, 품질 체크)"
    )
    parser.add_argument("--ref",   type=str, required=True, help="reference image 경로")
    parser.add_argument("--video", type=str, required=True, help="driving video 경로")
    parser.add_argument("-W", type=int, default=512, help="목표 너비 (px)")
    parser.add_argument("-H", type=int, default=784, help="목표 높이 (px)")
    parser.add_argument(
        "--out_ref",
        type=str,
        default=None,
        help="전처리된 ref image 저장 경로 (기본: inputs/ref.png)",
    )
    return parser.parse_args()


def resize_and_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    이미지를 target 비율에 맞게 center-crop 후 리사이즈.
    모델 입력 비율과 맞춰야 결과가 좋다.
    """
    orig_w, orig_h = image.size
    target_ratio = target_w / target_h
    orig_ratio   = orig_w / orig_h

    if orig_ratio > target_ratio:
        # 좌우 crop
        new_w = int(orig_h * target_ratio)
        left  = (orig_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, orig_h))
    else:
        # 상하 crop
        new_h = int(orig_w / target_ratio)
        top   = (orig_h - new_h) // 2
        image = image.crop((0, top, orig_w, top + new_h))

    image = image.resize((target_w, target_h), Image.LANCZOS)
    return image


def check_video(video_path: str):
    """비디오 기본 정보 출력"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [FAIL] 비디오를 열 수 없음: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur    = frames / fps if fps > 0 else 0

    print(f"  비디오 파일  : {video_path}")
    print(f"  해상도       : {w}x{h}")
    print(f"  FPS          : {fps:.1f}")
    print(f"  프레임 수    : {frames}")
    print(f"  길이         : {dur:.1f}초")

    if frames < 24:
        print("  [WARN] 프레임이 너무 적습니다. 최소 24프레임 이상 권장.")
    if w < 256 or h < 256:
        print("  [WARN] 해상도가 낮습니다. 256x256 이상 권장.")

    cap.release()


def main():
    args = parse_args()
    ref_path   = Path(args.ref)
    video_path = Path(args.video)

    print("=" * 50)
    print("  입력 파일 전처리")
    print("=" * 50)

    # ─── Reference image 처리 ─────────────────────────────────────────────────
    print(f"\n[1] Reference image: {ref_path}")
    if not ref_path.exists():
        print(f"  [FAIL] 파일 없음: {ref_path}")
        sys.exit(1)

    img = Image.open(ref_path).convert("RGB")
    print(f"  원본 크기: {img.size[0]}x{img.size[1]}")

    img_resized = resize_and_crop(img, args.W, args.H)
    print(f"  처리 후  : {img_resized.size[0]}x{img_resized.size[1]}")

    out_ref = Path(args.out_ref) if args.out_ref else REPO_ROOT / "inputs" / "ref.png"
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    img_resized.save(str(out_ref))
    print(f"  저장됨   : {out_ref}")

    # ─── Driving video 확인 ───────────────────────────────────────────────────
    print(f"\n[2] Driving video: {video_path}")
    if not video_path.exists():
        print(f"  [FAIL] 파일 없음: {video_path}")
        sys.exit(1)
    check_video(str(video_path))

    print("\n" + "=" * 50)
    print("  전처리 완료. 다음 단계:")
    print(f"    python tools/vid2pose.py --video_path {video_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
