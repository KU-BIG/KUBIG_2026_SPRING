"""
3d/download_weights.py
diffusion 가중치를 ./weights 에 받는다 (Champ inference 용)

2d/download_weights.py 와 같은 방식이며 대상만 Champ 로 바뀐다:
    - base SD1.5 는 live MIRROR로 교체 (runwayml/stable-diffusion-v1-5는 Hub에서
      내려감). 이 mirror가 막히면 아래 SD15_REPO만 다른 값으로 바꾸면 된다.
    - 전부 ./weights/ 아래에 받는다 (config.yaml의 weights_dir와 일치).
    - weight 파일 확장자는 자동 감지 (.safetensors 또는 .bin) 해서 하나만 받는다.
    - idempotent: 이미 있는 파일은 건너뛰므로 RunPod persistent volume 위에서
      재실행해도 비용이 들지 않는다.

주의: 여기서 받는 건 "생성(diffusion)"에 필요한 가중치뿐이다.
raw 영상 → guidance 전처리(SMPL/렌더)에 필요한 HMR2/detectron2/SMPL 모델은
별도 환경에서 Champ 의 scripts.pretrained_models.download 로 받는다 (preprocess.py 참고).

결과 레이아웃:
    weights/
    ├── stable-diffusion-v1-5/unet/{config.json, diffusion_pytorch_model.*}
    ├── sd-vae-ft-mse/{config.json, diffusion_pytorch_model.*}
    ├── image_encoder/{config.json, pytorch_model.*}
    └── champ/
        ├── denoising_unet.pth
        ├── reference_unet.pth
        ├── motion_module.pth
        ├── guidance_encoder_depth.pth
        ├── guidance_encoder_normal.pth
        ├── guidance_encoder_semantic_map.pth
        └── guidance_encoder_dwpose.pth
"""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# --- mirror가 막히면 이 줄만 바꾸면 됨 ------------------------------------------------
SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"   # runwayml/sd-v1-5의 mirror
# -------------------------------------------------------------------------------------

VAE_REPO = "stabilityai/sd-vae-ft-mse"
IMAGE_ENCODER_REPO = "lambdalabs/sd-image-variations-diffusers"
CHAMP_REPO = "fudan-generative-ai/champ"

WEIGHTS = Path("./weights")

CHAMP_CKPTS = [
    "denoising_unet.pth",
    "reference_unet.pth",
    "motion_module.pth",
    "guidance_encoder_depth.pth",
    "guidance_encoder_normal.pth",
    "guidance_encoder_semantic_map.pth",
    "guidance_encoder_dwpose.pth",
]


def _fetch(repo_id, filename, local_dir, subfolder=None):
    """파일이 없을 때만 다운로드하고, 로컬 경로를 반환한다"""
    dst = Path(local_dir) / (subfolder or "") / filename
    if dst.exists():
        return dst
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder,
                    local_dir=str(local_dir))
    return dst


def _fetch_first(repo_id, candidates, local_dir, subfolder=None):
    """repo에 존재하는 첫 번째 후보 파일명을 다운로드한다 (.safetensors를 먼저 시도하고
    없으면 .bin으로 fallback), weight 파일을 정확히 하나만 받기 위함"""
    # 이미 하나라도 있으면 스킵
    for name in candidates:
        if (Path(local_dir) / (subfolder or "") / name).exists():
            return
    last = None
    for name in candidates:
        try:
            _fetch(repo_id, name, local_dir, subfolder=subfolder)
            return
        except EntryNotFoundError as e:
            last = e
            continue
    raise RuntimeError(
        f"{repo_id}" + (f"/{subfolder}" if subfolder else "") + f" 에서 {candidates} 를 찾지 못함"
    ) from last


def prepare_base_model():
    print(f"base SD1.5 (mirror: {SD15_REPO}) 준비 중 ...")
    local = WEIGHTS / "stable-diffusion-v1-5"
    _fetch(SD15_REPO, "config.json", local, subfolder="unet")
    _fetch_first(SD15_REPO,
                 ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"],
                 local, subfolder="unet")


def prepare_vae():
    print("VAE 준비 중 ...")
    local = WEIGHTS / "sd-vae-ft-mse"
    _fetch(VAE_REPO, "config.json", local)
    _fetch_first(VAE_REPO,
                 ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"],
                 local)


def prepare_image_encoder():
    print("image encoder 준비 중 ...")
    _fetch(IMAGE_ENCODER_REPO, "config.json", WEIGHTS, subfolder="image_encoder")
    _fetch_first(IMAGE_ENCODER_REPO,
                 ["pytorch_model.safetensors", "pytorch_model.bin"],
                 WEIGHTS, subfolder="image_encoder")


def prepare_champ():
    print("Champ 체크포인트 준비 중 ...")
    local = WEIGHTS / "champ"
    for f in CHAMP_CKPTS:
        _fetch(CHAMP_REPO, f, local)


if __name__ == "__main__":
    prepare_base_model()
    prepare_vae()
    prepare_image_encoder()
    prepare_champ()
    print(f"\n생성용 가중치 준비 완료: {WEIGHTS.resolve()}")
    print("raw 영상 전처리를 하려면 preprocess.py 와 docs 참고 (SMPL/Blender 별도 환경).")
