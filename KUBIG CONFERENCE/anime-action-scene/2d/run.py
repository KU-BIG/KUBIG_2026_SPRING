"""
2d/run.py
Moore-AnimateAnyone를 활용한 2D inference wrapper

입력: raw driving video, reference image를 입력 받아 2D character animation을 생성

실행 방법
     python run.py --ref character.png --video driving.mp4 --out result.mp4

     
필수 옵션:
    --ref       캐릭터 기준 이미지
    --video     따라 할 동작 영상
    --out       결과 영상 저장 경로

실험 옵션:
    --noise     초기 latent noise 방식 선택
                선택 가능: iid, repeat, freenoise
                기본값: iid

    -L          사용할 프레임 수
                기본값: 24

    --steps     denoising step 수
                기본값: 30

    --seed      랜덤 시드
                기본값: 42

영상 설정 옵션:
    -W          결과 영상 너비
                기본값: 512

    -H          결과 영상 높이
                기본값: 784

    --cfg       classifier-free guidance scale
                기본값: 3.5

    --fps       결과 영상 fps
                기본값: 입력 영상 fps 사용

디버그 옵션:
    --dump-pose 추출한 DWPose 스켈레톤 영상도 같이 저장
                결과 파일 이름 옆에 _pose 를 붙여 저장한다
                (노이즈 전략별로 포즈 입력이 같은지 눈으로 확인할 때 사용)

설정 파일:
    --config    config.yaml 경로
                기본값: 현재 폴더의 config.yaml

파이프라인:
    1. raw driving video 입력
    2. video frame 추출
    3. DWPose skeleton 추출
    4. reference image 입력
    5. Moore-AnimationAnyone 모델 로드
        -VAE
        -image encoder
        -reference_unet
        -pose_guider
        -denoising_unet + motion_module
        -scheduler
    6. initial latent noise 생성
        noise.py를 통해 iid / repeat / freenoise 중 하나 선택
    7. video 생성
    8. 결과 저장
        (옵션) --dump-pose 사용 시 3번의 skeleton도 별도 영상으로 저장

"""


import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
SUBMODULE = HERE / "Moore-AnimateAnyone"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SUBMODULE))


# config default 값
DEFAULTS = dict(
    weights_dir="weights",
    base_sd="stable-diffusion-v1-5",
    vae="sd-vae-ft-mse",
    image_encoder="image_encoder",
    motion_module="motion_module.pth",
    denoising_unet="denoising_unet.pth",
    reference_unet="reference_unet.pth",
    pose_guider="pose_guider.pth",
    inference_config="configs/inference/inference_v2.yaml",
    weight_dtype="fp16",
    width=512, height=784, length=24, steps=30, cfg=3.5, seed=42, fps=None,
    noise="iid",
    freenoise=dict(window_size=24, window_stride=4),
)


def load_config(path):
    """config.yaml 읽은 후, defaults에 덮어씌우기"""
    cfg = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        with open(p) as f:
            user = yaml.safe_load(f) or {}

        fn = dict(DEFAULTS["freenoise"])
        fn.update(user.get("freenoise") or {})
        cfg.update(user)
        cfg["freenoise"] = fn
    else:
        print(f"config 파일을 찾지 못했습니다: {path}")
    return cfg # 없으면 defaults 활용


def under(root: Path, name) -> str:
    """상대경로는 root 아래에서 찾기 / 절대경로는 그대로 사용"""
    p = Path(name)
    return str(p if p.is_absolute() else (root / p))


def fps_to_int(fps, fallback=8):
    """fps 값을 양의 정수로 변환"""
    try:
        value = int(round(float(fps)))
        return value if value > 0 else fallback
    except Exception:
        return fallback


def pose_dump_path(out_path):
    """--out 옆에 스켈레톤 영상 저장 경로를 만든다 (result.mp4 -> result_pose.mp4)"""
    p = Path(out_path)
    return str(p.with_name(f"{p.stem}_pose{p.suffix}"))


def save_result_video(video, out_path, fps):
    """생성된 video tensor를 결과 영상 파일로 저장한다.

    video shape: (1, C, F, H, W)
    value range: [0, 1]
    """
    from src.utils.util import save_videos_from_pil  # engine helper (encodes mp4/gif)

    v = video[0]  # (C, F, H, W)
    frames = []
    for i in range(v.shape[1]):
        frame = (v[:, i].clamp(0, 1) * 255).round().byte()   # (C, H, W)
        frame = frame.permute(1, 2, 0).cpu().numpy()          # (H, W, C)
        frames.append(Image.fromarray(frame))

    out_path = os.path.abspath(out_path)                      # ensure a non-empty dirname
    save_videos_from_pil(frames, out_path, fps=fps)
    return out_path


def save_pil_video(frames, out_path, fps):
    """PIL 이미지 리스트를 영상 파일로 저장"""
    from src.utils.util import save_videos_from_pil

    out_path = os.path.abspath(out_path)
    save_videos_from_pil(frames, out_path, fps=fps)
    return out_path


def parse_args():
    ap = argparse.ArgumentParser(description="Moore-AnimateAnyone 2D inference wrapper")
    ap.add_argument("--ref", required=True, help="reference 이미지 경로")
    ap.add_argument("--video", required=True, help="pose 영상 경로")
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
    ap.add_argument("-L", type=int, default=None, help="사용할 최대 프레임 수")
    ap.add_argument("--steps", type=int, default=None, help="denoising step 수")
    ap.add_argument("--cfg", type=float, default=None, help="classifier-free guidance scale")
    ap.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    ap.add_argument("--fps", type=int, default=None, help="결과 영상 fps")
    ap.add_argument("--dump-pose", action="store_true", help="DWPose 스켈레톤 영상도 함께 저장")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI > config > defaults
    W = args.W or cfg["width"]
    H = args.H or cfg["height"]
    L = args.L or cfg["length"]
    steps = args.steps or cfg["steps"]
    guidance = args.cfg if args.cfg is not None else cfg["cfg"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    strategy = args.noise or cfg["noise"]
    out_fps_cfg = args.fps if args.fps is not None else cfg["fps"]
    win = cfg["freenoise"]["window_size"]
    stride = cfg["freenoise"]["window_stride"]

    if not torch.cuda.is_available():
        print("[warn] CUDA not available — Moore-AnimateAnyone needs a GPU. "
              "Run this on RunPod/Colab. Continuing will almost certainly fail.")
    dt = torch.float16 if str(cfg["weight_dtype"]).lower() == "fp16" else torch.float32
    weights_root = (HERE / cfg["weights_dir"]) if not Path(cfg["weights_dir"]).is_absolute() \
        else Path(cfg["weights_dir"])

    for pth, what in [(args.ref, "ref image"), (args.video, "driving video")]:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"{what} not found: {pth}")

    # Point the engine's hard-coded DWPose onnx location at OUR weights/ layout.
    # (wholebody.py reads Path('./pretrained_weights')/'DWPose/*.onnx' at __init__ time.)
    import src.dwpose.wholebody as _wb
    _wb.ModelDataPathPrefix = weights_root

    # Heavy imports (after sys.path + dwpose patch).
    from diffusers import AutoencoderKL, DDIMScheduler
    from omegaconf import OmegaConf
    from transformers import CLIPVisionModelWithProjection
    from src.dwpose import DWposeDetector
    from src.models.pose_guider import PoseGuider
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from src.utils.util import get_fps, read_frames
    import noise as noise_strategies

    # -- [1/4] load weights (order fixed: base models first, then overwrite with trained) --
    print("[1/4] loading models ...")
    infer_cfg_path = under(SUBMODULE, cfg["inference_config"]) \
        if not Path(cfg["inference_config"]).is_absolute() else cfg["inference_config"]
    infer = OmegaConf.load(infer_cfg_path)

    vae = AutoencoderKL.from_pretrained(under(weights_root, cfg["vae"])).to("cuda", dtype=dt)
    reference_unet = UNet2DConditionModel.from_pretrained(
        under(weights_root, cfg["base_sd"]), subfolder="unet"
    ).to(dtype=dt, device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        under(weights_root, cfg["base_sd"]),
        under(weights_root, cfg["motion_module"]),
        subfolder="unet",
        unet_additional_kwargs=infer.unet_additional_kwargs,
    ).to(dtype=dt, device="cuda")
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=dt, device="cuda")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        under(weights_root, cfg["image_encoder"])
    ).to(dtype=dt, device="cuda")
    scheduler = DDIMScheduler(**OmegaConf.to_container(infer.noise_scheduler_kwargs))

    # overwrite with trained checkpoints (strict flags exactly as upstream)
    denoising_unet.load_state_dict(
        torch.load(under(weights_root, cfg["denoising_unet"]), map_location="cpu"), strict=False)
    reference_unet.load_state_dict(
        torch.load(under(weights_root, cfg["reference_unet"]), map_location="cpu"))
    pose_guider.load_state_dict(
        torch.load(under(weights_root, cfg["pose_guider"]), map_location="cpu"))

    pipe = Pose2VideoPipeline(
        vae=vae, image_encoder=image_enc, reference_unet=reference_unet,
        denoising_unet=denoising_unet, pose_guider=pose_guider, scheduler=scheduler,
    ).to("cuda", dtype=dt)

    # inject noise strategy (monkeypatch prepare_latents)
    noise_strategies.install(pipe, strategy, window_size=win, window_stride=stride)
    print(f"      noise strategy = {strategy}"
          + (f" (window_size={win}, window_stride={stride})" if strategy == "freenoise" else ""))

    # -- [2/4] DWPose skeletons from the RAW driving video --
    print("[2/4] extracting DWPose skeletons ...")
    detector = DWposeDetector().to("cuda")
    src_fps = get_fps(args.video)
    frames = read_frames(args.video)[:L]        # cap to requested length
    pose_list, confs = [], []
    for frame_pil in frames:
        skeleton_pil, score = detector(frame_pil)
        pose_list.append(skeleton_pil)
        confs.append(float(np.mean(score)))     # per-keypoint conf; evaluation is out of scope
    L_eff = len(pose_list)                        # latents & pose frames MUST match
    print(f"      {L_eff} frames @ src {src_fps} fps | mean keypoint conf ~ {np.mean(confs):.3f}")

    if strategy == "freenoise" and L_eff <= win:
        print(f"[warn] freenoise is a no-op when frames ({L_eff}) <= window_size ({win}); "
              f"output will equal iid. Increase -L / --length beyond {win} to see its effect.")

    # 결과 fps 확정 (pose 덤프와 최종 저장에서 함께 사용)
    out_fps = fps_to_int(out_fps_cfg) if out_fps_cfg else fps_to_int(src_fps)

    # (옵션) 추출한 스켈레톤도 영상으로 저장 -- 노이즈 전략별 포즈 입력 확인용
    if args.dump_pose:
        pose_saved = save_pil_video(pose_list, pose_dump_path(args.out), fps=out_fps)
        print(f"      pose skeletons -> {pose_saved}")

    # -- [3/4] generate (result only; no grid) --
    print("[3/4] generating ...")
    generator = torch.manual_seed(seed)
    ref_pil = Image.open(args.ref).convert("RGB")
    video = pipe(ref_pil, pose_list, W, H, L_eff, steps, guidance, generator=generator).videos

    # -- [4/4] save result-only video --
    print("[4/4] saving ...")
    saved = save_result_video(video, args.out, fps=out_fps)
    print(f"done -> {saved}")


if __name__ == "__main__":
    main()