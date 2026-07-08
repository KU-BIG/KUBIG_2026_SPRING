"""
Step 2: 드라이빙 비디오 → 포즈 비디오 변환
실행: python tools/vid2pose.py --video_path inputs/driving.mp4

출력: inputs/driving_kps.mp4
  - 각 프레임의 사람 관절(keypoint)을 DWPose로 검출해서
    stick figure 형태로 그린 비디오를 저장한다.
  - 이 포즈 비디오가 최종 inference의 motion 정보로 들어간다.

참고: DWPose는 ONNX 기반이라 GPU가 없어도 CPU로 동작 가능.
      단, GPU(onnxruntime-gpu) 있으면 훨씬 빠름.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ─── Moore-AnimateAnyone 레포 루트를 sys.path에 추가 ───────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from src.dwpose import DWposeDetector
from src.utils.util import get_fps, read_frames, save_videos_from_pil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DWPose keypoints from a driving video."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="경로: 포즈를 추출할 드라이빙 비디오 (.mp4)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="출력 경로 (기본: 입력 파일명_kps.mp4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="onnxruntime 실행 디바이스 (기본: 자동 감지, MPS는 cpu로 대체)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="처리할 최대 프레임 수 (기본: 전체)",
    )
    return parser.parse_args()


def extract_pose_from_video(
    video_path: str,
    output_path: str = None,
    device: str = "cuda",
    max_frames: int = None,
):
    """
    video_path: 입력 비디오 경로
    output_path: 출력 포즈 비디오 경로 (None이면 자동 생성)
    device: "cuda" or "cpu"
    max_frames: 처리할 최대 프레임 수 (None이면 전체)

    Returns: output_path (str)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일이 없음: {video_path}")

    if output_path is None:
        output_path = video_path.parent / (video_path.stem + "_kps.mp4")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input : {video_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")

    # DWPose 초기화
    detector = DWposeDetector()
    detector = detector.to(device)

    # 비디오 읽기
    fps = get_fps(str(video_path))
    frames = read_frames(str(video_path))
    if max_frames is not None:
        frames = frames[:max_frames]

    print(f"Total frames: {len(frames)}, FPS: {fps}")

    # 각 프레임에서 포즈 추출
    kps_results = []
    for i, frame_pil in enumerate(frames):
        result, score = detector(frame_pil)
        score_mean = np.mean(score, axis=-1)
        kps_results.append(result)
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"  [{i+1}/{len(frames)}] mean keypoint score: {score_mean:.3f}")

    # 포즈 비디오 저장
    save_videos_from_pil(kps_results, str(output_path), fps=fps)
    print(f"\nPose video saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    args = parse_args()
    # DWPose(ONNX)는 cuda/cpu만 지원 → MPS면 cpu로 대체
    import torch
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    extract_pose_from_video(
        video_path=args.video_path,
        output_path=args.output_path,
        device=device,
        max_frames=args.max_frames,
    )
