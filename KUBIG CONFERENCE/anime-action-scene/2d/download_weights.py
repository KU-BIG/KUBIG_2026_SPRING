"""
2d/download_weights.py
가중치를 ./weights (우리 레이아웃)로 받아오는 스크립트, inference 전용

upstream tools/download_weights.py와 다른 점:
    - base SD1.5 repo를 살아있는 MIRROR로 교체 (runwayml/stable-diffusion-v1-5는 Hub에서
      삭제됨). 이 mirror가 막히면 아래 SD15_REPO만 다른 값으로 바꾸면 된다.
    - 전부 ./pretrained_weights/ 가 아닌 ./weights/ 아래에 저장한다 (config.yaml과 일치).
    - weight 파일 확장자는 자동 감지 (.safetensors 또는 .bin) 해서 하나만 받는다.
    - idempotent: 이미 있는 파일은 건너뛰므로 RunPod persistent volume
      (예: /workspace/weights) 위에서 재실행해도 비용이 들지 않는다.

결과 레이아웃 (총 ~15GB):
    weights/
    ├── stable-diffusion-v1-5/unet/{config.json, diffusion_pytorch_model.*}
    ├── sd-vae-ft-mse/{config.json, diffusion_pytorch_model.*}
    ├── image_encoder/{config.json, pytorch_model.*}
    ├── DWPose/{yolox_l.onnx, dw-ll_ucoco_384.onnx}
    ├── denoising_unet.pth
    ├── reference_unet.pth
    ├── pose_guider.pth
    └── motion_module.pth
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
DWPOSE_REPO = "yzd-v/DWPose"
ANYONE_REPO = "patrolli/AnimateAnyone"

WEIGHTS = Path("./weights")


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
        f"none of {candidates} found in {repo_id}"
        + (f"/{subfolder}" if subfolder else "")) from last


def prepare_base_model():
    print(f"Preparing base SD1.5 weights from mirror: {SD15_REPO} ...")
    local = WEIGHTS / "stable-diffusion-v1-5"
    _fetch(SD15_REPO, "config.json", local, subfolder="unet")
    _fetch_first(SD15_REPO,
                 ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"],
                 local, subfolder="unet")


def prepare_vae():
    print("Preparing VAE weights ...")
    local = WEIGHTS / "sd-vae-ft-mse"
    _fetch(VAE_REPO, "config.json", local)
    _fetch_first(VAE_REPO,
                 ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"],
                 local)


def prepare_image_encoder():
    print("Preparing image encoder weights ...")
    # weights/image_encoder/ 에 저장됨 (repo의 subfolder 이름 == 우리 폴더 이름)
    _fetch(IMAGE_ENCODER_REPO, "config.json", WEIGHTS, subfolder="image_encoder")
    _fetch_first(IMAGE_ENCODER_REPO,
                 ["pytorch_model.safetensors", "pytorch_model.bin"],
                 WEIGHTS, subfolder="image_encoder")


def prepare_dwpose():
    print("Preparing DWPose onnx weights ...")
    local = WEIGHTS / "DWPose"
    for f in ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]:
        _fetch(DWPOSE_REPO, f, local)


def prepare_anyone():
    print("Preparing AnimateAnyone checkpoints ...")
    for f in ["denoising_unet.pth", "reference_unet.pth", "pose_guider.pth", "motion_module.pth"]:
        _fetch(ANYONE_REPO, f, WEIGHTS)


if __name__ == "__main__":
    prepare_base_model()
    prepare_vae()
    prepare_image_encoder()
    prepare_dwpose()
    prepare_anyone()
    print(f"\nAll weights ready under: {WEIGHTS.resolve()}")
