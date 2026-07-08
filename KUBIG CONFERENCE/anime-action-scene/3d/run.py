"""
3d/run.py
Champ 를 활용한 3D character animation inference wrapper

입력: reference image + SMPL guidance 로 3D-aware character animation 을 생성
      (2d 와 형식 동일: --ref --video --out. 단 --video 는 guidance 폴더 또는 raw 영상)


실행 방법
     # 사전 렌더된 guidance 폴더로:
     python run.py --ref character.png --video motions/motion-01 --out result.mp4

     # raw 영상으로 (SMPL/Blender 전처리 환경 필요, preprocess.py):
     python run.py --ref character.png --video dance.mp4 --out result.mp4


필수 옵션:
    --ref       캐릭터 기준 이미지
    --video     guidance 폴더 경로 (또는 raw 영상 -> 전처리 필요)
    --out       결과 영상 저장 경로

실험 옵션:
    --noise     초기 latent noise 방식 선택
                선택 가능: iid, repeat, freenoise
                기본값: iid

    -L          사용할 프레임 수 (frame_range 를 [0, L] 로 설정)
                기본값: config 의 frame_range

    --steps     denoising step 수
                기본값: 20

    --seed      랜덤 시드
                기본값: 42

영상 설정 옵션:
    -W          결과 영상 너비
                기본값: 512

    -H          결과 영상 높이
                기본값: 512

    --cfg       classifier-free guidance scale
                기본값: 3.5

    --fps       결과 영상 fps
                기본값: 24 (guidance 에는 원본 fps 정보가 없음)

설정 파일:
    --config    config.yaml 경로
                기본값: 현재 폴더의 config.yaml

파이프라인:
    1. reference image 입력
    2. guidance 준비
        - --video 가 폴더면: depth/normal/semantic_map/dwpose (+mask) 프레임 그대로 사용
        - --video 가 영상이면: SMPL 피팅 -> 렌더로 guidance 생성 (preprocess.py, 별도 환경)
    3. Champ 모델 로드
        -image encoder / VAE
        -reference_unet
        -denoising_unet + motion_module
        -guidance_encoder x4 (depth / normal / semantic_map / dwpose)
        -scheduler (zero-SNR, v-prediction)
    4. initial latent noise 생성
        noise.py를 통해 iid / repeat / freenoise 중 하나 선택
    5. video 생성
    6. ref 크기로 resize 후 결과 저장

"""


import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

HERE = Path(__file__).resolve().parent
SUBMODULE = HERE / "champ"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SUBMODULE))


def load_config(path):
    """config.yaml 을 OmegaConf 로 읽는다 (Champ 는 설정 블록이 많아 yaml 이 필수)"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config 파일이 필요합니다: {path}")
    return OmegaConf.load(str(p))


def under(root: Path, name) -> str:
    """상대경로는 root 아래에서 찾기 / 절대경로는 그대로 사용"""
    p = Path(name)
    return str(p if p.is_absolute() else (root / p))


def fps_to_int(fps, fallback=24):
    """fps 값을 양의 정수로 변환 (guidance 는 원본 fps 정보가 없어 기본 24)"""
    try:
        value = int(round(float(fps)))
        return value if value > 0 else fallback
    except Exception:
        return fallback


def resolve_guidance_source(video_arg, ref_path, frame_range):
    """--video 를 guidance 폴더 경로로 바꾼다.

    폴더면 그대로 사용 (사전 렌더된 SMPL guidance),
    영상 파일이면 preprocess 로 위임한다 (Blender + 4D-Humans + SMPL 모델; 별도 환경).
    """
    if os.path.isdir(video_arg):
        return os.path.abspath(video_arg)
    if not os.path.exists(video_arg):
        raise FileNotFoundError(f"--video 경로 없음: {video_arg}")

    print("[info] --video 가 영상 파일 -> guidance 전처리 시작 (무거운 별도 경로)")
    import preprocess  # 무거운 import 라 이 분기에서만 로드
    return preprocess.build_guidance(video_arg, ref_path, frame_range=frame_range)


def process_semantic_map(semantic_map_path: Path):
    """semantic_map 에 같은 이름의 mask 를 적용해 배경을 검게 만든다 (Champ 원본 로직)"""
    name = semantic_map_path.name
    mask_path = semantic_map_path.parent.parent / "mask" / name
    semantic = np.array(Image.open(semantic_map_path))
    mask = np.array(Image.open(mask_path).convert("RGB"))
    return Image.fromarray(np.where(mask > 0, semantic, 0))


def load_guidance_group(guidance_dir, guidance_types, frame_range):
    """guidance 폴더에서 타입별 프레임 PIL 리스트를 읽는다.

    폴더 구조: guidance_dir/{depth,normal,semantic_map,dwpose}/*.png (+ mask/*.png)
    반환: ({type: [PIL, ...]}, 프레임 수). 모든 타입의 프레임 수는 같아야 한다.
    """
    group = {}
    for gtype in guidance_types:
        sub = Path(guidance_dir) / gtype
        if not sub.is_dir():
            raise FileNotFoundError(f"guidance 하위 폴더 없음: {sub}")
        files = sorted(sub.iterdir())
        if frame_range:
            files = files[frame_range[0]:frame_range[1]]
        if gtype == "semantic_map":
            group[gtype] = [process_semantic_map(p) for p in files]
        else:
            group[gtype] = [Image.open(p).convert("RGB") for p in files]

    length = len(next(iter(group.values())))
    if not all(len(v) == length for v in group.values()):
        raise ValueError("guidance 타입별 프레임 수가 다릅니다")
    return group, length


def build_guidance_encoders(cfg, dt):
    """guidance_types 별 GuidanceEncoder 생성 (depth / normal / semantic_map / dwpose)"""
    from models.guidance_encoder import GuidanceEncoder

    group = {}
    for gtype in cfg.guidance_types:
        group[gtype] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        ).to(device="cuda", dtype=dt)
    return group


def save_result_video(video, out_path, fps):
    """생성된 video tensor 를 결과 영상 파일로 저장한다 (grid 없이 결과만).

    video shape: (1, C, F, H, W)
    value range: [0, 1]
    """
    from utils.video_utils import save_videos_from_pil  # engine helper (encodes mp4/gif)

    v = video[0]  # (C, F, H, W)
    frames = []
    for i in range(v.shape[1]):
        frame = (v[:, i].clamp(0, 1) * 255).round().byte()   # (C, H, W)
        frame = frame.permute(1, 2, 0).cpu().numpy()          # (H, W, C)
        frames.append(Image.fromarray(frame))

    out_path = os.path.abspath(out_path)                      # ensure a non-empty dirname
    save_videos_from_pil(frames, out_path, fps=fps)
    return out_path


def parse_args():
    ap = argparse.ArgumentParser(description="Champ 3D inference wrapper")
    ap.add_argument("--ref", required=True, help="reference 이미지 경로")
    ap.add_argument("--video", required=True, help="guidance 폴더 경로 (또는 raw 영상 -> 전처리)")
    ap.add_argument("--out", required=True, help="result 저장 경로")
    ap.add_argument(
        "--noise",
        choices=list(("iid", "repeat", "freenoise")),
        default=None,
        help="초기 latent noise 방식 선택: iid, repeat, freenoise",
    )
    ap.add_argument("--config", default=str(HERE / "config.yaml"), help="config.yaml 경로")
    ap.add_argument("-W", type=int, default=None, help="결과 영상 너비")
    ap.add_argument("-H", type=int, default=None, help="결과 영상 높이")
    ap.add_argument("-L", type=int, default=None, help="사용할 최대 프레임 수 (frame_range 를 [0, L] 로)")
    ap.add_argument("--steps", type=int, default=None, help="denoising step 수")
    ap.add_argument("--cfg", type=float, default=None, help="classifier-free guidance scale")
    ap.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    ap.add_argument("--fps", type=int, default=None, help="결과 영상 fps")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI > config
    W = args.W or cfg.width
    H = args.H or cfg.height
    steps = args.steps or cfg.num_inference_steps
    guidance = args.cfg if args.cfg is not None else cfg.guidance_scale
    seed = args.seed if args.seed is not None else cfg.seed
    strategy = args.noise or cfg.get("noise", "iid")
    out_fps_cfg = args.fps if args.fps is not None else cfg.get("fps", None)
    win = cfg.freenoise.window_size
    stride = cfg.freenoise.window_stride
    frame_range = [0, args.L] if args.L else (
        list(cfg.data.frame_range) if cfg.data.get("frame_range") else None)

    if not torch.cuda.is_available():
        print("[warn] CUDA 없음 — Champ 는 GPU 가 필요. RunPod/Colab 에서 실행할 것. "
              "이대로 진행하면 거의 실패한다.")
    dt = torch.float16 if str(cfg.weight_dtype).lower() == "fp16" else torch.float32
    weights_root = (HERE / cfg.weights_dir) if not Path(cfg.weights_dir).is_absolute() \
        else Path(cfg.weights_dir)

    if not os.path.exists(args.ref):
        raise FileNotFoundError(f"ref image not found: {args.ref}")

    # 가중치 경로를 절대경로로 (실행 위치와 무관하게 동작)
    cfg.base_model_path = under(weights_root, cfg.base_model_path)
    cfg.vae_model_path = under(weights_root, cfg.vae_model_path)
    cfg.image_encoder_path = under(weights_root, cfg.image_encoder_path)
    cfg.ckpt_dir = under(weights_root, cfg.ckpt_dir)
    cfg.motion_module_path = under(weights_root, cfg.motion_module_path)

    # --video: 폴더면 guidance 그대로, 영상이면 전처리(무거운 별도 경로)로 위임
    guidance_dir = resolve_guidance_source(args.video, args.ref, frame_range)

    # heavy engine imports (after sys.path setup)
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection
    from models.unet_2d_condition import UNet2DConditionModel
    from models.unet_3d import UNet3DConditionModel
    from models.mutual_self_attention import ReferenceAttentionControl
    from models.champ_model import ChampModel
    from pipelines.pipeline_aggregation import MultiGuidance2LongVideoPipeline
    from utils.video_utils import resize_tensor_frames
    import noise as noise_strategies

    # -- [1/4] load models + build pipeline (order fixed: base first, then overwrite with trained) --
    print("[1/4] loading models ...")
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    scheduler = DDIMScheduler(**sched_kwargs)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path).to(dtype=dt, device="cuda")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(dtype=dt, device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=cfg.unet_additional_kwargs,
    ).to(dtype=dt, device="cuda")
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path, subfolder="unet").to(device="cuda", dtype=dt)
    guidance_encoder_group = build_guidance_encoders(cfg, dt)

    # overwrite with trained checkpoints (strict flags exactly as upstream: all strict=False)
    denoising_unet.load_state_dict(
        torch.load(osp.join(cfg.ckpt_dir, "denoising_unet.pth"), map_location="cpu"), strict=False)
    reference_unet.load_state_dict(
        torch.load(osp.join(cfg.ckpt_dir, "reference_unet.pth"), map_location="cpu"), strict=False)
    for gtype, module in guidance_encoder_group.items():
        module.load_state_dict(
            torch.load(osp.join(cfg.ckpt_dir, f"guidance_encoder_{gtype}.pth"), map_location="cpu"),
            strict=False)

    # reference attention 연결 (writer=reference_unet / reader=denoising_unet)
    reference_control_writer = ReferenceAttentionControl(
        reference_unet, do_classifier_free_guidance=False, mode="write", fusion_blocks="full")
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet, do_classifier_free_guidance=False, mode="read", fusion_blocks="full")
    model = ChampModel(
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader,
        guidance_encoder_group=guidance_encoder_group,
    ).to("cuda", dtype=dt)

    pipe_guidance = {
        f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}")
        for g in cfg.guidance_types
    }
    pipe = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=model.reference_unet,
        denoising_unet=model.denoising_unet,
        **pipe_guidance,
        scheduler=scheduler,
        guidance_process_size=cfg.data.get("guidance_process_size", None),
    ).to("cuda", dt)

    # inject noise strategy (monkeypatch prepare_latents; identical to 2d)
    noise_strategies.install(pipe, strategy, window_size=win, window_stride=stride)
    print(f"      noise strategy = {strategy}"
          + (f" (window_size={win}, window_stride={stride})" if strategy == "freenoise" else ""))

    # -- [2/4] load guidance (pre-rendered SMPL guidance) --
    print("[2/4] loading guidance ...")
    guidance_group, L_eff = load_guidance_group(guidance_dir, list(cfg.guidance_types), frame_range)
    print(f"      {L_eff} frames | types = {', '.join(cfg.guidance_types)} | from {guidance_dir}")

    if strategy == "freenoise" and L_eff <= win:
        print(f"[warn] freenoise is a no-op when frames ({L_eff}) <= window_size ({win}); "
              f"output will equal iid. Increase -L / frame_range beyond {win} to see its effect.")

    # 결과 fps 확정 (guidance 에는 원본 fps 정보가 없음)
    out_fps = fps_to_int(out_fps_cfg) if out_fps_cfg else fps_to_int(None)

    # -- [3/4] generate (result only; no grid) --
    print("[3/4] generating ...")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    ref_pil = Image.open(args.ref)
    ref_w, ref_h = ref_pil.size
    video = pipe(
        ref_pil, guidance_group, W, H, L_eff,
        num_inference_steps=steps, guidance_scale=guidance, generator=generator,
    ).videos

    # -- [4/4] save result-only video (resize to ref size, matches upstream animation.mp4) --
    print("[4/4] saving ...")
    video = resize_tensor_frames(video, (ref_h, ref_w))
    saved = save_result_video(video, args.out, fps=out_fps)
    print(f"done -> {saved}")


if __name__ == "__main__":
    main()
