# Moore-AnimateAnyone 구현 가이드

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/goldkangsan/moore-animate-anyone/blob/main/colab_animate_anyone.ipynb)

## 전체 실행 흐름

```
[사람 이미지 1장]  +  [동작 비디오]
       ↓                    ↓
  prepare_inputs.py    vid2pose.py (포즈 추출)
       ↓                    ↓
            pose2vid.py (inference)
                   ↓
           output/*.mp4 (결과 영상)
```

---

## 사전 준비

### 1. 레포 클론 (원본 Moore-AnimateAnyone)
```bash
git clone https://github.com/MooreThreads/Moore-AnimateAnyone
cd Moore-AnimateAnyone
```

> **주의**: 이 폴더의 스크립트는 Moore-AnimateAnyone 레포 루트에서 실행해야 한다.
> `src/`, `configs/` 등이 원본 레포에 있다.

### 2. 환경 설치
```bash
# Python 3.10+, CUDA 11.7+ 권장
conda create -n animate python=3.10 -y
conda activate animate
pip install -r requirements.txt
```

### 3. Weights 다운로드
```bash
python tools/download_weights.py
```

다운받는 파일들:
```
pretrained_weights/
├── stable-diffusion-v1-5/unet/
├── image_encoder/
├── sd-vae-ft-mse/
├── DWPose/
│   ├── dw-ll_ucoco_384.onnx
│   └── yolox_l.onnx
├── denoising_unet.pth
├── reference_unet.pth
├── pose_guider.pth
└── motion_module.pth
```

---

## 실행 순서

### Step 0: 환경 체크 (먼저 해라)
```bash
python tools/check_env.py
```

### Step 1: 입력 파일 전처리
```bash
# ref.png를 512x784로 리사이즈해서 inputs/ref.png로 저장
python tools/prepare_inputs.py \
  --ref path/to/your_photo.jpg \
  --video path/to/driving.mp4 \
  -W 512 -H 784
```

### Step 2: 포즈 추출
```bash
python tools/vid2pose.py --video_path inputs/driving.mp4
# → outputs/pose/driving_kps.mp4 생성
```

### Step 3: animation.yaml 수정
`configs/prompts/animation.yaml`:
```yaml
test_cases:
  "./inputs/ref.png":
    - "./outputs/pose/driving_kps.mp4"
```

### Step 4: Inference 실행
```bash
# 기본 (512x784, 32프레임)
python scripts/pose2vid.py \
  --config ./configs/prompts/animation.yaml \
  -W 512 -H 784 -L 32

# VRAM 부족할 때 (해상도/프레임 낮추기)
python scripts/pose2vid.py \
  --config ./configs/prompts/animation.yaml \
  -W 384 -H 512 -L 16
```

### (선택) Gradio UI로 실행
```bash
python run_gradio.py
# → http://localhost:7860
```

### 한 번에 전부 실행
```bash
bash run.sh                  # 기본 설정
bash run.sh --low_vram       # VRAM 부족 시
bash run.sh --frames 64      # 더 긴 영상
```

---

## VRAM 가이드

| 설정 | 최소 VRAM |
|------|-----------|
| 384x512, 16프레임 | ~8GB |
| 512x784, 32프레임 | ~12GB |
| 512x784, 64프레임 | ~16GB+ |

---

## 자주 막히는 포인트

### onnxruntime 오류
```bash
pip install onnxruntime-gpu==1.16.3
# GPU 없으면:
pip install onnxruntime==1.16.3
```

### CUDA out of memory
- `-L` (프레임 수) 줄이기
- `-W`, `-H` 해상도 줄이기
- `weight_dtype: fp16` 확인

### weights 경로 오류
- `pretrained_weights/` 폴더 구조가 README 기준과 정확히 맞아야 함
- `python tools/check_env.py` 로 확인

### pose 추출이 안 될 때
- DWPose onnx 파일 확인: `pretrained_weights/DWPose/*.onnx`
- 비디오 codec 문제: mp4 (H.264) 형식 사용 권장

---

## 파일 구조

```
animate_anyone/
├── scripts/
│   └── pose2vid.py          # 메인 inference 스크립트
├── tools/
│   ├── download_weights.py  # weights 다운로드
│   ├── vid2pose.py          # 포즈 추출
│   ├── prepare_inputs.py    # 입력 전처리
│   └── check_env.py         # 환경 체크
├── configs/
│   ├── prompts/
│   │   └── animation.yaml   # 입력/출력 경로 설정
│   └── inference/
│       └── inference_v2.yaml # UNet/scheduler 설정
├── inputs/                  # ref image 저장 위치
├── outputs/
│   └── pose/                # vid2pose 출력 위치
├── pretrained_weights/      # 모델 weights
├── run.sh                   # 원클릭 실행 스크립트
└── run_gradio.py            # Gradio UI
```
