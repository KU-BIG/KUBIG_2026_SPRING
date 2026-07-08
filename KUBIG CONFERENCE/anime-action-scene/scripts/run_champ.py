"""
Champ inference wrapper — 3D SMPL 가이드 기반 캐릭터 애니메이션 생성

실행:
  # SD 1.5 기반 (비교용 baseline)
  python scripts/run_champ.py --config configs/champ/inference_3060.yaml

  # Anything V5 기반 (애니 특화)
  python scripts/run_champ.py --config configs/champ/inference_3060_anime.yaml

사전 준비:
  1. python tools/download_weights.py --champ
  2. python scripts/preprocess_smpl.py --video inputs/action.mp4
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# ─── Champ 레포를 sys.path에 추가 ─────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
CHAMP_ROOT = REPO_ROOT / "champ"
sys.path.insert(0, str(CHAMP_ROOT))
sys.path.insert(0, str(REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from scripts.pose2vid import get_device, get_weight_dtype


def load_champ_pipeline(cfg, device: str, weight_dtype: torch.dtype):
    """Champ 모델 컴포넌트 로드"""
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection
    from models.unet_2d_condition import UNet2DConditionModel
    from models.unet_3d import UNet3DConditionModel
    from models.guidance_encoder import GuidanceEncoder
    from pipelines.pipeline_pose2vid_long import MultiGuidance2LongVideoPipeline

    print("[1/5] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae"
    ).to(device, dtype=weight_dtype)

    print("[2/5] Loading CLIP Image Encoder...")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path
    ).to(device, dtype=weight_dtype)

    print("[3/5] Loading Reference UNet (2D)...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet"
    ).to(device, dtype=weight_dtype)

    print("[4/5] Loading Denoising UNet (3D) + Motion Module...")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.pretrained_model_name_or_path,
        cfg.champ_pretrained_weight_path.motion_module,
        subfolder="unet",
    ).to(device, dtype=weight_dtype)

    print("[5/5] Loading Guidance Encoders (depth/normal/semantic/dwpose)...")
    guidance_encoders = {}
    for g_type in cfg.guidance_types:
        enc = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_embedding_channels,
            guidance_input_channels=3,
            block_out_channels=(16, 32, 96, 256),
        ).to(device, dtype=weight_dtype)
        ckpt_key = f"guidance_encoder_{g_type}"
        ckpt_path = cfg.champ_pretrained_weight_path[ckpt_key]
        enc.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        guidance_encoders[g_type] = enc

    # ─── pretrained weights 로드 ──────────────────────────────────────────────
    w = cfg.champ_pretrained_weight_path
    denoising_unet.load_state_dict(torch.load(w.denoising_unet, map_location="cpu"), strict=False)
    reference_unet.load_state_dict(torch.load(w.reference_unet, map_location="cpu"))
    print("All weights loaded.\n")

    # ─── 메모리 최적화 ────────────────────────────────────────────────────────
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="linear", clip_sample=False,
        num_train_timesteps=1000, steps_offset=1,
        prediction_type="epsilon",
    )
    pipe = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        guidance_encoder_group=guidance_encoders,
        scheduler=scheduler,
    ).to(device)

    if cfg.get("xformers_memory_efficient_attention", False):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xformers enabled")
        except Exception:
            print("[WARN] xformers 사용 불가 (무시하고 계속)")

    if cfg.get("enable_attention_slicing", False):
        pipe.enable_attention_slicing()
        print("Attention slicing enabled")

    return pipe


def run_champ_inference(pipe, cfg, device: str, seed: int):
    """Champ 추론 실행 → 결과 영상 저장"""
    from PIL import Image
    from datetime import datetime
    import importlib

    # Champ의 combine_guidance_data 유틸 사용
    try:
        combine = importlib.import_module("inference").combine_guidance_data
    except Exception:
        combine = None

    ref_image = Image.open(cfg.ref_image_path).convert("RGB")
    guidance_folder = Path(cfg.guidance_folder)

    # 가이드 데이터 로드
    guidance_data = {}
    for g_type in cfg.guidance_types:
        g_dir = guidance_folder / g_type
        if g_dir.exists():
            frames = sorted(g_dir.glob("*.png"))[:cfg.num_frames]
            guidance_data[g_type] = [Image.open(f).convert("RGB") for f in frames]
        else:
            print(f"[WARN] 가이드 폴더 없음: {g_dir} — 해당 가이드 스킵")

    generator = torch.Generator()
    generator.manual_seed(seed)

    print(f"Running Champ inference...")
    print(f"  Frames: {cfg.num_frames} | {cfg.width}x{cfg.height}")
    print(f"  Steps : {cfg.num_inference_steps} | CFG: {cfg.guidance_scale}")

    result = pipe(
        ref_image=ref_image,
        guidance_images=guidance_data,
        width=cfg.width,
        height=cfg.height,
        num_frames=cfg.num_frames,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
    )

    # 결과 저장
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"output_{time_str}.mp4"

    from champ.utils.video_utils import save_video
    save_video(result.videos, str(out_path), fps=8)
    print(f"\n  Saved: {out_path}")
    return str(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Champ 3D 기반 캐릭터 애니메이션 생성")
    parser.add_argument("--config", required=True, help="Champ config yaml 경로")
    parser.add_argument("--ref_image",      default=None, help="참조 이미지 경로 (config 오버라이드)")
    parser.add_argument("--guidance_folder",default=None, help="가이드 폴더 경로 (config 오버라이드)")
    parser.add_argument("--seed", type=int,  default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)

    if args.ref_image:       cfg.ref_image_path  = args.ref_image
    if args.guidance_folder: cfg.guidance_folder = args.guidance_folder
    if args.seed:            cfg.seed            = args.seed

    device       = get_device()
    weight_dtype = get_weight_dtype(device)

    print("=" * 60)
    print("Champ: 3D SMPL-guided Character Animation")
    print(f"  Config  : {args.config}")
    print(f"  Device  : {device} | dtype: {weight_dtype}")
    print(f"  Base    : {cfg.pretrained_model_name_or_path}")
    print("=" * 60)

    pipe     = load_champ_pipeline(cfg, device, weight_dtype)
    out_path = run_champ_inference(pipe, cfg, device, seed=cfg.get("seed", 42))

    print("\n" + "=" * 60)
    print(f"Done: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
