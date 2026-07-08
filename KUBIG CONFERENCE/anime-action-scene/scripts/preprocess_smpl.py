"""
전처리 파이프라인 — 드라이빙 영상 → Champ 입력 가이드 데이터 생성 (맥북 호환)

실행:
  python scripts/preprocess_smpl.py --video inputs/action.mp4

출력 폴더 구조:
  outputs/guidance/
    ├── dwpose/     2D 스켈레톤 (ONNX 기반, CPU 동작)
    ├── depth/      깊이 맵    (MiDaS, MPS/CPU 동작)
    ├── normal/     법선 맵    (depth에서 계산, CPU)
    └── semantic/   시맨틱 맵  (MediaPipe, CPU 동작)

의존성:
  pip install torch torchvision opencv-python Pillow tqdm
  pip install mediapipe          # semantic 맵용
  # MiDaS는 torch.hub로 자동 다운로드됨
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─── 디바이스 감지 ─────────────────────────────────────────────────────────────
def get_device(prefer: str = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── 프레임 추출 ──────────────────────────────────────────────────────────────
def extract_frames(video_path: str, out_dir: Path, max_frames: int = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(total, max_frames) if max_frames else total

    frame_paths = []
    for i in tqdm(range(n), desc="프레임 추출"):
        ret, frame = cap.read()
        if not ret:
            break
        path = out_dir / f"{i:04d}.png"
        cv2.imwrite(str(path), frame)
        frame_paths.append(str(path))
    cap.release()
    print(f"  {len(frame_paths)}프레임 추출 완료 @ {fps:.1f} fps")
    return frame_paths, fps


# ─── DWPose (2D 스켈레톤) ──────────────────────────────────────────────────────
def run_dwpose(frame_paths: list, out_dir: Path):
    """
    DWPose는 ONNX 기반 → CPU/GPU 모두 동작. 맥북 OK.
    tools/vid2pose.py 의 DWposeDetector 재사용.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from src.dwpose import DWposeDetector
        detector = DWposeDetector()
        detector = detector.to("cpu")  # ONNX는 CPU로 충분
        for i, fp in enumerate(tqdm(frame_paths, desc="DWPose 추출")):
            img_pil = Image.open(fp).convert("RGB")
            result, _ = detector(img_pil)
            result.save(out_dir / f"{i:04d}.png")
        print(f"  DWPose 완료 → {out_dir}")
    except ImportError:
        print("[WARN] DWPose 로드 실패. pretrained_weights/DWPose 가중치 확인 필요.")


# ─── Depth 맵 (MiDaS) ─────────────────────────────────────────────────────────
def run_midas_depth(frame_paths: list, out_dir: Path, device: str):
    """
    MiDaS small 모델 → MPS/CPU 모두 동작. 맥북 OK.
    torch.hub으로 자동 다운로드 (~100MB).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  MiDaS 모델 로드 중 (device: {device})...")

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    midas.to(device).eval()

    for i, fp in enumerate(tqdm(frame_paths, desc="Depth 추출")):
        img = cv2.imread(fp)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).to(device)

        with torch.no_grad():
            depth = midas(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_np = depth.cpu().numpy()
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out_dir / f"{i:04d}.png"), depth_colored)

    print(f"  Depth 완료 → {out_dir}")


# ─── Normal 맵 (Depth에서 계산) ───────────────────────────────────────────────
def compute_normal_maps(depth_dir: Path, out_dir: Path):
    """
    Depth 맵에서 법선 맵 계산. 순수 numpy 연산 → CPU만으로 OK.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_files = sorted(depth_dir.glob("*.png"))

    for fp in tqdm(depth_files, desc="Normal 계산"):
        depth = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)

        normal = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = (normal / (norm + 1e-6) * 127.5 + 127.5).astype(np.uint8)

        cv2.imwrite(str(out_dir / fp.name), cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))

    print(f"  Normal 완료 → {out_dir}")


# ─── Semantic 맵 (MediaPipe) ───────────────────────────────────────────────────
def run_semantic_segmentation(frame_paths: list, out_dir: Path):
    """
    MediaPipe Selfie Segmentation → CPU 동작. 맥북 OK.
    머리/몸통/배경을 색상으로 구분한 맵 생성.
    pip install mediapipe
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import mediapipe as mp
        seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        # Champ 시맨틱 색상 테이블
        PERSON_COLOR = np.array([100, 200, 100], dtype=np.uint8)   # 사람 → 초록
        BG_COLOR     = np.array([0,   0,   0  ], dtype=np.uint8)   # 배경 → 검정

        for i, fp in enumerate(tqdm(frame_paths, desc="Semantic 추출")):
            img = cv2.imread(fp)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = seg.process(img_rgb)
            mask = result.segmentation_mask > 0.5

            semantic = np.where(mask[:, :, None], PERSON_COLOR, BG_COLOR)
            cv2.imwrite(str(out_dir / f"{i:04d}.png"), cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR))

        seg.close()
        print(f"  Semantic 완료 → {out_dir}")

    except ImportError:
        print("[WARN] mediapipe 미설치. pip install mediapipe")
        print("  Semantic 맵 없이 계속합니다 (depth/normal/dwpose만 사용)")


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def preprocess(video_path: str, out_dir: str, max_frames: int, width: int, height: int):
    device = get_device()
    out = Path(out_dir)

    print("=" * 60)
    print("전처리: 드라이빙 영상 → Champ 가이드 데이터")
    print(f"  입력  : {video_path}")
    print(f"  출력  : {out}")
    print(f"  프레임: 최대 {max_frames}장 | {width}x{height}")
    print(f"  디바이스: {device}")
    print("=" * 60)

    # 1. 프레임 추출
    frames_dir = out / "_frames"
    frame_paths, fps = extract_frames(video_path, frames_dir, max_frames)

    # 2. DWPose
    print("\n[1/4] DWPose (2D 스켈레톤)...")
    run_dwpose(frame_paths, out / "dwpose")

    # 3. Depth (MiDaS)
    print("\n[2/4] Depth 맵 (MiDaS)...")
    run_midas_depth(frame_paths, out / "depth", device)

    # 4. Normal (Depth에서 계산)
    print("\n[3/4] Normal 맵...")
    compute_normal_maps(out / "depth", out / "normal")

    # 5. Semantic (MediaPipe)
    print("\n[4/4] Semantic 맵 (MediaPipe)...")
    run_semantic_segmentation(frame_paths, out / "semantic")

    print("\n" + "=" * 60)
    print(f"완료! 가이드 데이터 저장 위치: {out}")
    print(f"  {out}/dwpose/    — 2D 스켈레톤")
    print(f"  {out}/depth/     — 깊이 맵")
    print(f"  {out}/normal/    — 법선 맵")
    print(f"  {out}/semantic/  — 시맨틱 맵")
    print("\n다음 단계 (Colab에서 실행):")
    print(f"  python scripts/run_champ.py --config configs/champ/inference_3060_anime.yaml")
    print("=" * 60)
    return str(out)


def parse_args():
    parser = argparse.ArgumentParser(description="드라이빙 영상 → Champ 가이드 데이터 추출 (맥북 호환)")
    parser.add_argument("--video",      required=True,              help="드라이빙 영상 경로")
    parser.add_argument("--out",        default="outputs/guidance", help="출력 폴더")
    parser.add_argument("--max_frames", type=int, default=16,       help="최대 프레임 수 (기본: 16)")
    parser.add_argument("--width",      type=int, default=384)
    parser.add_argument("--height",     type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.video, args.out, args.max_frames, args.width, args.height)
