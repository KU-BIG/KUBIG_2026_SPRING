# 3d/ — Champ 3D Inference Wrapper

reference 이미지 + SMPL guidance(사전 렌더) 로 3D-parametric 기반 character animation 을
생성한다. [Champ][champ] 엔진(`champ/` submodule) 위에 올린 얇은 wrapper이며,
**학습/파인튜닝 없음** — 공개 가중치를 그대로 쓴다. 계약(CLI)은 `2d/` 와 동일하다.

입력: reference 이미지, guidance 폴더(또는 raw 영상 + 전처리)
출력: 생성된 character animation video (ref/pose/result grid가 아닌 결과 영상만)

## 폴더 구조

```
3d/
├── run.py                 # 진입점 (이걸 실행)
├── noise.py               # 초기 latent noise 전략: iid / repeat / freenoise (2d 와 동일 파일)
├── config.yaml            # 경로 + 생성 설정 + noise 선택 (Champ 설정 구조)
├── download_weights.py    # 생성용 가중치를 ./weights 로 받음 (SD1.5 미러 + Champ ckpt)
├── preprocess.py          # (무거운 별도 경로) raw 영상 -> guidance 생성 오케스트레이터
├── requirements.txt       # inference 의존성 (torch 는 별도 설치)
├── .gitignore             # weights/ 결과물은 커밋 안 함
└── champ/                 # 엔진 submodule (models / pipelines / data_processors)
```

## 계약

`2d/`(Moore) 와 동일한 CLI — 통합 시 얇은 디스패처 한 장이면 된다.

```bash
python run.py --ref <character.png> --video <guidance_dir | raw_video.mp4> --out <out.mp4> \
              [--noise iid|repeat|freenoise] [--config config.yaml] \
              [-W 512] [-H 512] [-L 100] [--steps 20] [--cfg 3.5] [--seed 42] [--fps 24]
```

결과는 **생성된 영상만** 저장한다(ref 크기로 resize). CLI 플래그가 `config.yaml` 을 덮어쓴다.

> **2d 와 다른 핵심**: Champ 의 `--video` 는 raw 영상이 아니라 **SMPL guidance** 를 먹는다.
> 아래 두 경로 중 하나로 guidance 를 준비한다.

## guidance 준비 (둘 중 하나)

**경로 A — 사전 렌더된 guidance 폴더 (권장, 가볍고 바로 됨)**

Champ 가 제공하는 예시 모션을 받거나, 이미 렌더해 둔 폴더를 쓴다. `--video` 에 그 폴더를 준다.

```bash
git clone https://huggingface.co/datasets/fudan-generative-ai/champ_motions_example example_data
python run.py --ref character.png --video example_data/motions/motion-01 --out result.mp4
```

guidance 폴더 구조(프레임별 PNG):

```
motion-01/
├── depth/         0000.png ...
├── normal/        0000.png ...
├── semantic_map/  0000.png ...
├── dwpose/        0000.png ...
└── mask/          0000.png ...   # semantic_map 배경 처리에 사용
```

**경로 B — raw 영상에서 직접 생성 (무거운 별도 환경)**

`--video` 에 영상 파일을 주면 `run.py` 가 자동으로 `preprocess.py` 를 호출한다. 단, 이 경로는
**Blender 3.6 + 4D-Humans + detectron2 + SMPL body model(.pkl, 라이선스 필요) + DWPose 리포**가
있어야 하며, 이 저장소에서 실행 검증되지 않았다. 설치는 `champ/docs/data_process.md` 참고.

```bash
python run.py --ref character.png --video dance.mp4 --out result.mp4
```

## 셋업

NVIDIA GPU 필요. `3d/` 안에서:

```bash
# 0. 엔진 (submodule 로 없으면)
git clone https://github.com/fudan-generative-vision/champ
#   또는: git submodule update --init --recursive

# 1. torch 먼저 (CUDA 에 맞게). RunPod PyTorch 이미지면 이미 있음 -> 건너뜀
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# 2. 나머지
pip install -r requirements.txt

# 3. 생성용 가중치 -> ./weights
python download_weights.py
```

SD1.5 미러가 막히면 `download_weights.py` 상단 `SD15_REPO` 만 바꾸면 된다.

## 실행 방법

```bash
python run.py --ref character.png --video example_data/motions/motion-01 --out result.mp4
python run.py --ref character.png --video example_data/motions/motion-01 --out result_fn.mp4 --noise freenoise -L 48
```

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
    -W          결과 영상 너비 (기본값: 512)
    -H          결과 영상 높이 (기본값: 512)
    --cfg       classifier-free guidance scale (기본값: 3.5)
    --fps       결과 영상 fps (기본값: 24, guidance 에는 원본 fps 정보가 없음)

설정 파일:
    --config    config.yaml 경로 (기본값: 현재 폴더의 config.yaml)

## 노이즈 전략 (`--noise`)

초기 latent 만 조작한다(학습 불필요). latent 는 `(B, C, F, H/8, W/8)` 이고, 프레임 축 `F` 를
채우는 방식만 다르며, 이후 표준 `* scheduler.init_noise_sigma` 스케일이 붙는다. **`prepare_latents`
가 Moore 와 완전히 동일해서 `2d/noise.py` 와 같은 파일을 그대로 쓴다.**

| 전략        | 방식                                | 특징                          |
|-------------|-------------------------------------|-------------------------------|
| `iid`       | 프레임마다 독립 노이즈 (기본)       | 움직임 자유, 깜빡임 가능       |
| `repeat`    | 한 프레임 노이즈를 전 프레임에 복사 | 일관성 최대, 움직임 억제       |
| `freenoise` | 윈도우 재사용 + 지역 셔플           | 긴 영상 일관성, 학습 불필요    |

파이프라인의 `prepare_latents` 를 몽키패치해 주입한다. `freenoise` 기본값(`window_size: 24`,
`window_stride: 4`)은 파이프라인 context window(`context_frames=24`, `context_overlap=4`)에 맞췄다.

> **freenoise 주의**: 프레임 수가 `window_size` 보다 **커야** `iid` 와 달라진다. 기본 `frame_range`
> 나 `-L` 로 24 이하만 쓰면 재스케줄 루프가 비어 `iid` 와 동일하다. `-L 48` 이상으로 확인할 것.

## 가중치 및 라이선스

`2d/weights/` 와 `3d/weights/` 는 **완전히 별개**다. Moore 와 Champ 둘 다
`denoising_unet.pth` / `reference_unet.pth` / `motion_module.pth` 라는 **같은 파일명**을 쓰지만
**내용이 다르다** — 절대 섞지 말 것.

`download_weights.py` 는 "생성"용 가중치만 받는다(SD1.5 미러, VAE, image encoder, Champ ckpt).
raw 영상 전처리용 HMR2/detectron2/SMPL 모델은 별도(경로 B, `docs/data_process.md`).

코드는 Apache-2.0([Champ][champ]; wrapper 코드는 우리 것). 가중치 라이선스는 코드와 별개이며,
특히 SD1.5 는 CreativeML OpenRAIL-M, **SMPL body model 은 별도 라이선스**다. 크레딧 체인:
Champ, (원조) AnimateAnyone, AnimateDiff, DWPose, SMPL, Stable Diffusion. 배포 전 각 모델
카드에서 최종 저작자 표기를 확인할 것.
