"""
Step 1: pretrained weights 다운로드 스크립트
실행:
  python tools/download_weights.py           # SD 1.5 기반 (기본)
  python tools/download_weights.py --anime   # Anything V5 추가 다운로드

다운받는 것들:
  - stable-diffusion-v1-5 (UNet)
  - image_encoder (CLIP)
  - DWPose (pose detector: yolox_l.onnx, dw-ll_ucoco_384.onnx)
  - sd-vae-ft-mse (VAE)
  - AnimateAnyone weights (denoising_unet, reference_unet, pose_guider, motion_module)
  - [--anime] Anything V5 UNet (SD 1.5 호환 애니 특화 가중치)
"""

import argparse
import os
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parent.parent / "pretrained_weights"


def prepare_base_model():
    print(">>> [1/5] Preparing stable-diffusion-v1-5 UNet weights...")
    local_dir = BASE_DIR / "stable-diffusion-v1-5"
    local_dir.mkdir(parents=True, exist_ok=True)
    # v1-5-pruned-emaonly.ckpt (4GB) 은 inference에 불필요 → 제외
    files = [
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        "scheduler/scheduler_config.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "text_encoder/config.json",
        "text_encoder/pytorch_model.bin",
        "feature_extractor/preprocessor_config.json",
        "model_index.json",
    ]
    # runwayml/stable-diffusion-v1-5 는 deprecated → 신규 repo 우선, 실패 시 fallback
    repo_ids = [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "runwayml/stable-diffusion-v1-5",
    ]
    for hub_file in files:
        path = Path(hub_file)
        saved_path = local_dir / path
        if saved_path.exists():
            print(f"  [SKIP] {hub_file}")
            continue
        downloaded = False
        for repo_id in repo_ids:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    subfolder=str(PurePosixPath(path.parent)) if str(path.parent) != "." else None,
                    filename=path.name,
                    local_dir=str(local_dir),
                )
                print(f"  [OK] {hub_file}")
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            print(f"  [WARN] {hub_file} 다운 실패 — 수동 확인 필요")


def prepare_image_encoder():
    print(">>> [2/5] Preparing image_encoder (CLIP) weights...")
    local_dir = BASE_DIR
    local_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "image_encoder/config.json",
        "image_encoder/pytorch_model.bin",
    ]
    for hub_file in files:
        path = Path(hub_file)
        saved_path = local_dir / path
        if saved_path.exists():
            print(f"  [SKIP] {hub_file}")
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=str(PurePosixPath(path.parent)),
            filename=path.name,
            local_dir=str(local_dir),
        )
        print(f"  [OK] {hub_file}")


def prepare_dwpose():
    print(">>> [3/5] Preparing DWPose weights (ONNX)...")
    local_dir = BASE_DIR / "DWPose"
    local_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "dw-ll_ucoco_384.onnx",
        "yolox_l.onnx",
    ]
    for hub_file in files:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            print(f"  [SKIP] {hub_file}")
            continue
        hf_hub_download(
            repo_id="yzd-v/DWPose",
            filename=hub_file,
            local_dir=str(local_dir),
        )
        print(f"  [OK] {hub_file}")


def prepare_vae():
    print(">>> [4/5] Preparing VAE (sd-vae-ft-mse) weights...")
    local_dir = BASE_DIR / "sd-vae-ft-mse"
    local_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "config.json",
        "diffusion_pytorch_model.bin",
    ]
    for hub_file in files:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            print(f"  [SKIP] {hub_file}")
            continue
        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            filename=hub_file,
            local_dir=str(local_dir),
        )
        print(f"  [OK] {hub_file}")


def prepare_animate_anyone():
    print(">>> [5/5] Preparing AnimateAnyone pretrained weights...")
    local_dir = BASE_DIR
    local_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "denoising_unet.pth",
        "motion_module.pth",
        "pose_guider.pth",
        "reference_unet.pth",
    ]
    for hub_file in files:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            print(f"  [SKIP] {hub_file}")
            continue
        hf_hub_download(
            repo_id="patrolli/AnimateAnyone",
            filename=hub_file,
            local_dir=str(local_dir),
        )
        print(f"  [OK] {hub_file}")


def prepare_anything_v5():
    """
    Anything V5 UNet 다운로드 (SD 1.5 UNet 교체용).
    HuggingFace repo: Linaqruf/anything-v5.0
    SD 1.5와 동일한 UNet 아키텍처 → VAE/motion_module/pose_guider는 그대로 재사용 가능.
    """
    print(">>> [6/6] Preparing Anything V5 UNet (anime base model)...")
    local_dir = BASE_DIR / "anything-v5"
    local_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        "scheduler/scheduler_config.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "text_encoder/config.json",
        "text_encoder/pytorch_model.bin",
        "feature_extractor/preprocessor_config.json",
        "model_index.json",
    ]
    for hub_file in files:
        path = Path(hub_file)
        saved_path = local_dir / path
        if saved_path.exists():
            print(f"  [SKIP] {hub_file}")
            continue
        try:
            hf_hub_download(
                repo_id="Linaqruf/anything-v5.0",
                subfolder=str(PurePosixPath(path.parent)) if str(path.parent) != "." else None,
                filename=path.name,
                local_dir=str(local_dir),
            )
            print(f"  [OK] {hub_file}")
        except Exception as e:
            print(f"  [WARN] {hub_file} 다운 실패: {e}")


def prepare_champ_weights():
    """
    Champ pretrained 가중치 다운로드.
    HuggingFace: fudan-generative-vision/champ
    포함: denoising_unet, reference_unet, motion_module, guidance_encoder_*
    """
    print(">>> [7/7] Preparing Champ pretrained weights (3D pipeline)...")
    local_dir = BASE_DIR / "champ"
    local_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "denoising_unet.pth",
        "reference_unet.pth",
        "motion_module.pth",
        "guidance_encoder_depth.pth",
        "guidance_encoder_normal.pth",
        "guidance_encoder_semantic.pth",
        "guidance_encoder_dwpose.pth",
    ]
    for hub_file in files:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            print(f"  [SKIP] {hub_file}")
            continue
        try:
            hf_hub_download(
                repo_id="fudan-generative-vision/champ",
                filename=hub_file,
                local_dir=str(local_dir),
            )
            print(f"  [OK] {hub_file}")
        except Exception as e:
            print(f"  [WARN] {hub_file} 다운 실패: {e}")


def check_weights():
    """다운로드된 weights 상태 확인"""
    print("\n>>> Checking pretrained_weights/ structure...\n")
    required = [
        BASE_DIR / "stable-diffusion-v1-5" / "unet" / "config.json",
        BASE_DIR / "stable-diffusion-v1-5" / "unet" / "diffusion_pytorch_model.bin",
        BASE_DIR / "image_encoder" / "config.json",
        BASE_DIR / "image_encoder" / "pytorch_model.bin",
        BASE_DIR / "DWPose" / "dw-ll_ucoco_384.onnx",
        BASE_DIR / "DWPose" / "yolox_l.onnx",
        BASE_DIR / "sd-vae-ft-mse" / "config.json",
        BASE_DIR / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin",
        BASE_DIR / "denoising_unet.pth",
        BASE_DIR / "reference_unet.pth",
        BASE_DIR / "pose_guider.pth",
        BASE_DIR / "motion_module.pth",
    ]
    all_ok = True
    for p in required:
        status = "[OK]" if p.exists() else "[MISSING]"
        if not p.exists():
            all_ok = False
        rel = p.relative_to(BASE_DIR.parent)
        print(f"  {status} {rel}")
    print()
    if all_ok:
        print("All weights ready. You can run inference.\n")
    else:
        print("Some weights are missing. Re-run this script or download manually.\n")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anime", action="store_true",
                        help="Anything V5 UNet 다운로드 (애니 스타일용)")
    parser.add_argument("--champ", action="store_true",
                        help="Champ 3D 파이프라인 가중치 다운로드")
    args = parser.parse_args()

    print("=" * 60)
    print("Anime Action Scene: Downloading pretrained weights")
    print("=" * 60)
    prepare_base_model()
    prepare_image_encoder()
    prepare_dwpose()
    prepare_vae()
    prepare_animate_anyone()

    if args.anime:
        prepare_anything_v5()

    if args.champ:
        prepare_champ_weights()

    check_weights()
