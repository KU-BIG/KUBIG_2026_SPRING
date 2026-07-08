> 📦 **전체 코드·커밋 히스토리**: [goldkangsan/anime-action-scene](https://github.com/goldkangsan/anime-action-scene) · **KUBIG 아카이브**: [KU-BIG/2026_SPRING-Anime_Action_Scene](https://github.com/KU-BIG/2026_SPRING-Anime_Action_Scene)
>
> 컨퍼런스 제출용 스냅샷입니다. 엔진(submodule) 코드는 위 원본 repo를 참고하세요.

# Anime Action Scene Generator
**KUBIG Conference · Multimodal Team**

> 캐릭터 이미지 1장 + 동작 영상 → 그 캐릭터가 그 동작을 하는 애니메이션 영상 생성

세대별 SOTA character animation 모델을 **직접 재현·비교**하고 애니 도메인에 적용한 프로젝트.
**학습/파인튜닝 없이** 공개 가중치를 그대로 불러와 inference만 수행한다.

---

## 프로젝트 개요 — "동작을 어떻게 주는가"로 세대 비교

| 세대 | 모델 | 동작 가이드 | 이 레포 |
|---|---|---|---|
| **2D** (2023) | AnimateAnyone (Moore) | DWPose 2D 스켈레톤 | ✅ [`2d/`](2d/README.md) |
| **3D** (2024) | Champ | SMPL 3D (depth·normal·semantic·dwpose) | ✅ [`3d/`](3d/README.md) |
| **최신** (2026) | UniAnimate | DWPose + video diffusion 백본 | ✅ [`unianimate/`](unianimate/README.md) |

**공통 구조** (세 모델 공통): 확산(Diffusion) 기반 ·
**ReferenceNet**(외형 identity) + **Pose Guider**(동작 주입) + **Denoising UNet** + **Motion Module**(프레임 간 일관성).
→ 세대 차이는 결국 **동작을 표현하는 방식(가이드)**의 차이다.

---

## 핵심 발견

- **3D SMPL 가이드는 정보(깊이·표면)가 풍부하지만, 프레임 간 떨림(jitter)이 있다.**
- training-free 노이즈 재배치(FreeNoise)로도 이 떨림은 크게 줄지 않았다 → **문제는 노이즈가 아니라 입력 가이드 자체.**
- 최신 UniAnimate가 SMPL(3D)을 버리고 DWPose(2D)로 회귀한 흐름과 일치. → **정보량 ≠ 안정성.**

> ⚠️ **방법론 주의**: FreeNoise는 프레임 수 `L`이 `window_size`(기본 24)보다 **클 때만** `iid`와 달라진다.
> `-L 24` 이하로 실험하면 재배치 루프가 비어 결과가 `iid`와 동일하다. 효과를 보려면 `-L 48` 이상으로 확인할 것.

---

## 결과 영상

같은 캐릭터(이누야샤)로 세대별 결과를 비교했다 — 분할 화면 = **참조 · 생성 · 가이드맵**.

| 영상 | 모델 | 구성 | 포인트 |
|---|---|---|---|
| [`champ_inuyasha.mp4`](results/champ_inuyasha.mp4) | 3D Champ | ref · 생성 · dwpose · depth · normal | SMPL 가이드맵 4종이 함께 보임 |
| [`unianimate_inuyasha.mp4`](results/unianimate_inuyasha.mp4) | 최신 UniAnimate | ref · dwpose · 생성 | DWPose 기반, 셋 중 가장 깔끔 |
| [`champ_freenoise_inuyasha.mp4`](results/champ_freenoise_inuyasha.mp4) | 3D Champ + FreeNoise | 5분할 | FreeNoise 적용해도 기본 대비 변화 미미 |

> GitHub에서 파일명을 클릭하면 브라우저에서 바로 재생된다.

---

## 폴더 구조

```
anime-action-scene/
├── 2d/                      # Moore-AnimateAnyone 2D wrapper   → 2d/README.md
│   ├── run.py               #   진입점
│   ├── noise.py             #   training-free 노이즈 전략 (iid/repeat/freenoise)
│   ├── config.yaml
│   └── Moore-AnimateAnyone/  #  엔진 submodule
├── 3d/                      # Champ 3D wrapper                 → 3d/README.md
│   ├── run.py
│   ├── noise.py             #   2d와 동일 파일 (prepare_latents 몽키패치)
│   ├── preprocess.py        #   raw 영상 → SMPL guidance (무거운 별도 경로)
│   └── champ/                #  엔진 submodule
├── unianimate/              # UniAnimate 최신 모델 wrapper      → unianimate/README.md
│   ├── UniAnimate_infer_my.yaml
│   └── UniAnimate/           #  엔진 submodule
├── results/                 # 세대별 결과 영상 (이누야샤)
├── run_gradio.py            # Gradio 데모 UI
├── colab_2d_baseline.ipynb  # Colab 실험 노트북
├── colab_champ_inference.ipynb
├── colab_animate_anyone.ipynb
├── configs/                 # 2D/3D inference 설정
└── README_KR.md             # 2D 상세 실행 가이드
```

> 실행법·CLI 옵션·가중치·라이선스는 각 폴더 README에 상세히 있다:
> **[2d/README.md](2d/README.md)** · **[3d/README.md](3d/README.md)**

---

## 빠른 시작

**2D (Moore-AnimateAnyone)**
```bash
cd 2d
python download_weights.py                                   # 가중치 ~15GB → ./weights
python run.py --ref character.png --video driving.mp4 --out result.mp4
python run.py --ref character.png --video driving.mp4 --out result_fn.mp4 --noise freenoise -L 48
```

**3D (Champ)** — `--video`에 raw 영상이 아니라 **SMPL guidance 폴더**를 준다
```bash
cd 3d
python download_weights.py
python run.py --ref character.png --video example_data/motions/motion-01 --out result.mp4
```

**최신 (UniAnimate)** — modelscope 가중치 + pose 정렬 후 inference (자세히는 [`unianimate/README.md`](unianimate/README.md))
```bash
cd unianimate/UniAnimate
python run_align_pose.py --ref_name data/images/ref.jpg --source_video_paths data/videos/clip.mp4 --saved_pose_dir data/saved_pose/ref_clip
python inference.py --cfg configs/UniAnimate_infer_my.yaml
```

**Gradio 데모**: `python run_gradio.py`

---

## Colab (설치 없이 실행)

### 2D Baseline (Moore-AnimateAnyone)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/goldkangsan/anime-action-scene/blob/main/colab_2d_baseline.ipynb)

### 3D Champ
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/goldkangsan/anime-action-scene/blob/main/colab_champ_inference.ipynb)

**실행 순서 (공통)**
```
셀 0 → 셀 1 → 셀 2 (pip 설치, 자동 재시작)
→ 셀 0 → 셀 1 → 셀 3 (import 테스트) → 이후 순서대로
```
> ⚠️ 셀 2 이후 런타임이 자동 재시작됨 — 정상 동작. 셀 2만 건너뛰고 다시 실행.

---

## training-free 노이즈 전략 (`--noise`)

초기 latent noise `(B, C, F, H/8, W/8)`의 프레임 축 `F`를 채우는 방식만 교체한다.
**재학습 없음** — `noise.py`가 pipeline의 `prepare_latents`를 몽키패치해서 주입한다.

| 전략 | 방식 | 특징 |
|---|---|---|
| `iid` (기본) | 프레임마다 독립 노이즈 | 움직임 자유, flicker 가능 |
| `repeat` | 한 프레임 노이즈를 전체 프레임에 복사 | 일관성 최대, 움직임 억제 |
| `freenoise` | 윈도우 재사용 + 지역 셔플 | 긴 영상 일관성, tuning 불필요 (`-L 48`↑에서 효과) |

---

## 검증된 패키지 버전

```
diffusers==0.24.0
transformers>=4.36.0,<4.40.0
huggingface_hub>=0.19.0,<0.23.0   # 0.23+ 에서 cached_download 삭제됨
accelerate
```

---

## 크레딧 · 라이선스

재현 기반: **Moore-AnimateAnyone · (원조) AnimateAnyone · Champ · AnimateDiff · DWPose · SMPL · Stable Diffusion**.
코드는 Apache-2.0(wrapper는 우리 것). 가중치 라이선스는 코드와 별개다 —
특히 SD1.5는 CreativeML OpenRAIL-M, SMPL body model은 별도 라이선스. 배포 전 각 모델 카드에서 최종 조건을 확인할 것.
