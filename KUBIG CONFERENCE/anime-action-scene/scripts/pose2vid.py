"""
Step 3: Reference Image + Pose Video → Animation 생성 (core inference)
실행:
  python scripts/pose2vid.py \\
    --config ./configs/prompts/animation.yaml \\
    -W 512 -H 784 -L 32

주요 파라미터:
  -W / -H  : 출력 해상도 (가로 x 세로). VRAM 부족하면 낮춰라.
  -L       : 생성할 프레임 수 (길수록 느리고 메모리 많이 씀)
  --seed   : 랜덤 시드 (같은 시드 = 같은 결과)
  --cfg    : classifier-free guidance scale (기본 3.5, 높을수록 포즈에 충실)
  --steps  : DDIM denoising steps (기본 30, 낮추면 빠르지만 품질 저하)

출력: output/YYYYMMDD/HHMM--seed_XX-WxH/ 아래 mp4 저장
     mp4에는 ref image / pose video / generated video 3열로 저장됨
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image
from einops import repeat
from omegaconf import OmegaConf
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection

# ─── 레포 루트를 sys.path에 추가 ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Moore-AnimateAnyone: character animation inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/prompts/animation.yaml",
        help="animation.yaml 경로",
    )
    parser.add_argument("-W", type=int, default=512,  help="출력 너비 (px)")
    parser.add_argument("-H", type=int, default=784,  help="출력 높이 (px)")
    parser.add_argument("-L", type=int, default=32,   help="생성 프레임 수")
    parser.add_argument("--seed",  type=int,   default=42,  help="랜덤 시드")
    parser.add_argument("--cfg",   type=float, default=3.5, help="CFG scale")
    parser.add_argument("--steps", type=int,   default=30,  help="DDIM steps")
    parser.add_argument("--fps",   type=int,   default=None, help="출력 FPS (기본: 입력 FPS 유지)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="실행 디바이스 (기본: 자동 감지)",
    )
    return parser.parse_args()


def get_device(prefer: str = None) -> str:
    """cuda → mps → cpu 순서로 자동 감지. prefer로 강제 지정 가능."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_weight_dtype(device: str) -> torch.dtype:
    """MPS는 일부 fp16 연산이 불안정하므로 fp32 사용."""
    if device == "mps":
        return torch.float32
    return torch.float16


def load_models(config, weight_dtype: torch.dtype, device: str):
    """모든 모델 컴포넌트를 로드하고 반환"""

    print("[1/5] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(
        device, dtype=weight_dtype
    )

    print("[2/5] Loading Reference UNet (2D)...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)

    print("[3/5] Loading Denoising UNet (3D) + Motion Module...")
    infer_config = OmegaConf.load(config.inference_config)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    print("[4/5] Loading Pose Guider...")
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )

    print("[5/5] Loading CLIP Image Encoder...")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    # ─── pretrained weights 로드 ─────────────────────────────────────────────
    print("\nLoading pretrained weights...")
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )
    print("All weights loaded.\n")

    return vae, reference_unet, denoising_unet, pose_guider, image_enc, infer_config


def build_pipeline(
    vae, image_enc, reference_unet, denoising_unet, pose_guider, infer_config,
    weight_dtype: torch.dtype, device: str
) -> Pose2VideoPipeline:
    """파이프라인 조립"""
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)
    return pipe


def run_inference(
    pipe: Pose2VideoPipeline,
    ref_image_path: str,
    pose_video_path: str,
    width: int,
    height: int,
    n_frames: int,
    steps: int,
    cfg: float,
    generator: torch.Generator,
    save_dir: Path,
    fps_override: int = None,
):
    """
    단일 (ref_image, pose_video) 쌍에 대해 inference 실행 후 mp4 저장.
    """
    ref_name  = Path(ref_image_path).stem
    pose_name = Path(pose_video_path).stem.replace("_kps", "")

    print(f"\n  ref  : {ref_image_path}")
    print(f"  pose : {pose_video_path}")

    # ─── 입력 로드 ───────────────────────────────────────────────────────────
    ref_image_pil = Image.open(ref_image_path).convert("RGB")

    pose_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])

    pose_images = read_frames(pose_video_path)
    src_fps = get_fps(pose_video_path)
    print(f"  pose video: {len(pose_images)} frames @ {src_fps} fps")

    pose_list        = []
    pose_tensor_list = []
    for pose_pil in pose_images[:n_frames]:
        pose_tensor_list.append(pose_transform(pose_pil))
        pose_list.append(pose_pil)

    # ─── 텐서 구성 ───────────────────────────────────────────────────────────
    ref_tensor = pose_transform(ref_image_pil)          # (C, H, W)
    ref_tensor = ref_tensor.unsqueeze(1).unsqueeze(0)   # (1, C, 1, H, W)
    ref_tensor = repeat(
        ref_tensor, "b c f h w -> b c (repeat f) h w", repeat=n_frames
    )

    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (F, C, H, W)
    pose_tensor = pose_tensor.transpose(0, 1)           # (C, F, H, W)
    pose_tensor = pose_tensor.unsqueeze(0)              # (1, C, F, H, W)

    # ─── inference 실행 ──────────────────────────────────────────────────────
    print("  Running diffusion inference...")
    video = pipe(
        ref_image_pil,
        pose_list,
        width,
        height,
        n_frames,
        steps,
        cfg,
        generator=generator,
    ).videos

    # ─── 저장 ────────────────────────────────────────────────────────────────
    # 3열 그리드로 저장: [reference | pose | generated]
    video_grid = torch.cat([ref_tensor, pose_tensor, video], dim=0)
    time_str   = datetime.now().strftime("%H%M")
    out_path   = save_dir / f"{ref_name}_{pose_name}_{height}x{width}_{int(cfg)}_{time_str}.mp4"

    save_videos_grid(
        video_grid,
        str(out_path),
        n_rows=3,
        fps=fps_override if fps_override is not None else src_fps,
    )
    print(f"  Saved: {out_path}")
    return str(out_path)


def main():
    args   = parse_args()
    config = OmegaConf.load(args.config)

    device       = get_device(args.device)
    weight_dtype = get_weight_dtype(device)
    if config.weight_dtype == "fp32":
        weight_dtype = torch.float32

    print("=" * 60)
    print("Moore-AnimateAnyone: Character Animation Inference")
    print(f"  Resolution : {args.W}x{args.H}")
    print(f"  Frames     : {args.L}")
    print(f"  Device     : {device}  |  dtype: {weight_dtype}")
    print(f"  Seed       : {args.seed}")
    print("=" * 60)

    # ─── 모델 로드 ───────────────────────────────────────────────────────────
    vae, ref_unet, denoise_unet, pose_guider, image_enc, infer_config = load_models(
        config, weight_dtype, device
    )
    pipe = build_pipeline(
        vae, image_enc, ref_unet, denoise_unet, pose_guider,
        infer_config, weight_dtype, device
    )

    # MPS는 device-specific generator가 불안정 → CPU generator 사용
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # ─── 출력 디렉토리 ───────────────────────────────────────────────────────
    date_str  = datetime.now().strftime("%Y%m%d")
    time_str  = datetime.now().strftime("%H%M")
    save_dir  = Path(f"output/{date_str}/{time_str}--seed_{args.seed}-{args.W}x{args.H}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {save_dir}\n")

    # ─── test_cases 순회 ─────────────────────────────────────────────────────
    results = []
    for ref_image_path, pose_video_paths in config["test_cases"].items():
        for pose_video_path in pose_video_paths:
            out = run_inference(
                pipe=pipe,
                ref_image_path=ref_image_path,
                pose_video_path=pose_video_path,
                width=args.W,
                height=args.H,
                n_frames=args.L,
                steps=args.steps,
                cfg=args.cfg,
                generator=generator,
                save_dir=save_dir,
                fps_override=args.fps,
            )
            results.append(out)

    print("\n" + "=" * 60)
    print(f"Done. {len(results)} video(s) generated:")
    for r in results:
        print(f"  {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
