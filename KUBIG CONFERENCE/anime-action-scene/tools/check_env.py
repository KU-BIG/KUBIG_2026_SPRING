"""
환경 체크 스크립트 — inference 실행 전에 먼저 돌려봐라.
실행: python tools/check_env.py

체크 항목:
  1. Python / CUDA / torch 버전
  2. 필수 패키지 import
  3. pretrained weights 존재 여부
  4. GPU 메모리 확인
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "pretrained_weights"

REQUIRED_WEIGHTS = [
    WEIGHTS_DIR / "stable-diffusion-v1-5" / "unet" / "config.json",
    WEIGHTS_DIR / "stable-diffusion-v1-5" / "unet" / "diffusion_pytorch_model.bin",
    WEIGHTS_DIR / "image_encoder" / "config.json",
    WEIGHTS_DIR / "image_encoder" / "pytorch_model.bin",
    WEIGHTS_DIR / "DWPose" / "dw-ll_ucoco_384.onnx",
    WEIGHTS_DIR / "DWPose" / "yolox_l.onnx",
    WEIGHTS_DIR / "sd-vae-ft-mse" / "config.json",
    WEIGHTS_DIR / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin",
    WEIGHTS_DIR / "denoising_unet.pth",
    WEIGHTS_DIR / "reference_unet.pth",
    WEIGHTS_DIR / "pose_guider.pth",
    WEIGHTS_DIR / "motion_module.pth",
]

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "einops",
    "omegaconf",
    "PIL",
    "cv2",
    "numpy",
    "av",
    "onnxruntime",
    "gradio",
    "accelerate",
    "decord",
    "imageio",
    "scipy",
    "skimage",
]


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def check_python():
    section("Python 버전")
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    status = "OK" if ok else "WARN (Python 3.10+ 권장)"
    print(f"  Python {v.major}.{v.minor}.{v.micro}  [{status}]")


def check_torch():
    section("PyTorch / CUDA")
    try:
        import torch
        print(f"  torch       : {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    GPU {i}: {name}  ({total:.1f} GB)")
        else:
            print("  [WARN] CUDA 없음 — CPU 모드로 실행됩니다 (매우 느림)")
    except ImportError:
        print("  [FAIL] torch not installed")


def check_packages():
    section("필수 패키지 import 체크")
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            print(f"  [OK]      {pkg}")
        except ImportError:
            print(f"  [MISSING] {pkg}  ← pip install {pkg}")


def check_weights():
    section("Pretrained Weights 존재 여부")
    all_ok = True
    for p in REQUIRED_WEIGHTS:
        exists = p.exists()
        if not exists:
            all_ok = False
        rel = p.relative_to(BASE_DIR)
        status = "OK     " if exists else "MISSING"
        print(f"  [{status}] {rel}")
    if not all_ok:
        print("\n  일부 weight 파일이 없습니다.")
        print("  실행: python tools/download_weights.py")
    else:
        print("\n  모든 weight 파일이 준비되었습니다.")


def check_inputs():
    section("입력 파일 확인")
    inputs_dir = BASE_DIR / "inputs"
    pose_dir   = BASE_DIR / "outputs" / "pose"

    ref_images = list(inputs_dir.glob("*.png")) + list(inputs_dir.glob("*.jpg"))
    pose_videos = list(pose_dir.glob("*_kps.mp4"))

    if ref_images:
        print("  Reference images:")
        for f in ref_images:
            print(f"    {f.name}")
    else:
        print("  [WARN] inputs/ 에 reference image가 없습니다.")
        print("         inputs/ref.png 파일을 준비해주세요.")

    if pose_videos:
        print("  Pose videos:")
        for f in pose_videos:
            print(f"    {f.name}")
    else:
        print("  [WARN] outputs/pose/ 에 포즈 비디오가 없습니다.")
        print("         python tools/vid2pose.py --video_path inputs/driving.mp4")


def main():
    print("=" * 50)
    print("  Moore-AnimateAnyone: Environment Check")
    print("=" * 50)
    check_python()
    check_torch()
    check_packages()
    check_weights()
    check_inputs()
    print("\n" + "=" * 50)
    print("  Check complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
