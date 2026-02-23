import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


# -----------------------------
# Config
# -----------------------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-scribble"  # scribble ControlNet
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# -----------------------------
# image loading
# -----------------------------
def load_image(path: str) -> np.ndarray:
    """
    이미지 파일을 불러와 grayscale numpy array(H, W)로 반환

    ControlNet scribble은 구조 정보만 필요하므로
    컬러 정보는 제거하고 grayscale로 변환
    """ 
    img = Image.open(path).convert("RGB")
    img = np.array(img)  # H,W,3 uint8
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def image_to_scribble(gray: np.ndarray, size: int = 512) -> Image.Image:
    """
    Grayscale 이미지를 ControlNet scribble 조건 이미지로 변환

    과정:
    1) Resize + padding → 512x512 정사각형
    2) Canny edge detection
    3) 흰 배경 + 검은 선 구조로 변환
    4) 선을 조금 두껍게 만들어 안정성 증가
    """
    # Resize first (preserve aspect by padding to square)
    h, w = gray.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size), 255, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized

    # Edge -> scribble
    edges = cv2.Canny(canvas, threshold1=50, threshold2=150)
    scribble = 255 - edges  # make background white, lines black

    # Thicken lines a bit (optional)
    kernel = np.ones((3, 3), np.uint8)
    scribble = cv2.erode(scribble, kernel, iterations=1)

    rgb = np.stack([scribble, scribble, scribble], axis=-1)
    return Image.fromarray(rgb)


# -----------------------------
# Stable Diffusion + ControlNet Pipeline
# -----------------------------
def build_pipe():
    # Scribble 구조를 이해하는 ControlNet 로드
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=DTYPE)
    # Stable Diffusion + ControlNet 결합
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,   # demo 목적이면 꺼도 됨 (환경 정책은 본인 환경에 맞게)
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)

    # Optional speed/memory tweaks
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe

# -----------------------------
# 이미지 Grid 생성 함수
# -----------------------------
def make_grid(images, cols: int = 3) -> Image.Image:
    """
    여러 이미지를 가로 방향 grid로 결합(CFG 비교용)
    """
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid


# -----------------------------
# Main experiment: CFG sweep
# -----------------------------
def run_cfg_sweep(
    img_path: str,
    prompt: str,
    negative_prompt: str = "low quality, blurry, deformed",
    cfg_list=(3.0, 7.0, 12.0),
    steps: int = 30,
    seed: int = 42,
):
    """
    동일한 조건 이미지에서
    CFG 값을 바꿔가며 생성 결과 비교

    CFG (Classifier-Free Guidance):
    - 낮으면 → 다양성 ↑ / 프롬프트 영향 ↓
    - 높으면 → 프롬프트 영향 ↑ / 과적합 위험
    """

    pipe = build_pipe()

    gray = load_image(img_path)
    control_img = image_to_scribble(gray, size=512)

    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    results = []
    for cfg in cfg_list:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_img,
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
            generator=gen,
        )
        results.append(out.images[0])

    stem = Path(img_path).stem
    control_img.save(OUT_DIR / f"control_{stem}.png")
    grid = make_grid(results, cols=len(results))
    grid.save(OUT_DIR / f"cfg_sweep_{stem}.png")

    print(f"[OK] saved: {OUT_DIR / f'cfg_sweep_{stem}.png'}")

# -----------------------------
# 실행 영역
# -----------------------------

if __name__ == "__main__":
    IMG_PATH = "house.JPG"
    PROMPT = "a small cozy house in a park, clear composition, simple illustration, friendly, high detail"
    NEG = "text, watermark, blurry, low quality, distorted, extra limbs"

    run_cfg_sweep(
        img_path=IMG_PATH,
        prompt=PROMPT,
        negative_prompt=NEG,
        cfg_list=(3.0, 7.0, 12.0),
        steps=30,
        seed=123,
    )