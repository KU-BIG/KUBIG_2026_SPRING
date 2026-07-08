# 2d/ — Moore-AnimateAnyone 2D Inference Wrapper

Moore-AnimateAnyone(`Moore-AnimateAnyone/` submodule) 기반 2D character animation
inference wrapper. **학습/파인튜닝은 하지 않고**, 공개된 학습 가중치를 그대로 불러와
reference 이미지 + driving video로부터 캐릭터 애니메이션을 생성한다.

입력: raw driving video, reference 이미지
출력: 생성된 character animation video (ref/pose/result grid가 아닌 결과 영상만)

## 폴더 구조

```
2d/
├── run.py                 # 진입점 (실행은 이 파일로)
├── noise.py                # 초기 latent noise 전략: iid / repeat / freenoise
├── config.yaml              # 가중치 경로 + 생성 옵션 + noise 선택
├── download_weights.py      # 가중치(~15GB)를 ./weights 로 받는 스크립트
├── requirements.txt          # inference 의존성 (torch는 별도 설치, 아래 참고)
├── .gitignore                # weights/, 결과 영상은 커밋하지 않음
└── Moore-AnimateAnyone/       # 엔진 submodule (models / pipeline / DWPose)
```

## 셋업

GPU(NVIDIA CUDA)가 필요하다. `2d/` 안에서 순서대로 진행:


# 1. torch를 먼저, 본인 버전에 맞게
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 2. 나머지 의존성
pip install -r requirements.txt

# 3. 가중치 -> ./weights (~15GB, idempotent이므로 재실행해도 안전)
python download_weights.py
```

SD1.5 mirror가 막혀 있으면 `download_weights.py` 상단의 `SD15_REPO`만 바꾸면 된다.

## 실행 방법

```bash
python run.py --ref character.png --video driving.mp4 --out result.mp4
python run.py --ref character.png --video driving.mp4 --out result_fn.mp4 --noise freenoise -L 48
```

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
    -W          결과 영상 너비 (기본값: 512)
    -H          결과 영상 높이 (기본값: 784)
    --cfg       classifier-free guidance scale (기본값: 3.5)
    --fps       결과 영상 fps (기본값: 입력 영상 fps 사용)

디버그 옵션:
    --dump-pose 추출한 DWPose 스켈레톤 영상도 같이 저장
                결과 파일 이름 옆에 `_pose` 를 붙여 저장한다 (예: `result.mp4` -> `result_pose.mp4`)
                (노이즈 전략별로 포즈 입력이 같은지 눈으로 확인할 때 사용)

설정 파일:
    --config    config.yaml 경로 (기본값: 현재 폴더의 config.yaml)

## 노이즈 전략 (`--noise`)

모두 초기 latent `(B, C, F, H/8, W/8)` 에만 관여하며, frame 축 `F`를 채우는 방식만
다르다. `noise.py`가 pipeline의 `prepare_latents`를 monkeypatch해서 주입한다.

| 전략        | 방식                                    | 특징                              |
|-------------|-----------------------------------------|-----------------------------------|
| `iid`       | 프레임마다 독립적인 noise (기본값)       | 움직임은 자유로우나 flicker 가능   |
| `repeat`    | 한 프레임의 noise를 모든 프레임에 복사   | 일관성 최대, 움직임 억제 가능      |
| `freenoise` | window 재사용 + local shuffle           | 긴 영상에서도 일관성 유지, tuning 불필요 |

> **주의:** `freenoise`는 프레임 수가 `window_size`(기본값 24)보다 커야 `iid`와
> 결과가 달라진다. 기본값 `-L 24`로는 재배치 루프가 비어서 결과가 `iid`와 동일하다.
> 실제 효과를 보려면 `-L 48`처럼 `window_size`보다 크게 지정할 것 — 이 경우 `run.py`가
> 경고를 출력한다.

## 가중치 및 라이선스

- 가중치는 총 ~15GB이며 저장소에 **커밋하지 않는다** (`.gitignore` 참고).
- `2d/weights/` 와 `3d/weights/` 는 서로 완전히 다르다. Moore와 Champ 둘 다
  `denoising_unet.pth` / `reference_unet.pth` / `motion_module.pth` 같은 동일한
  파일명을 쓰지만 **내용은 다르므로 절대 섞지 말 것**.
- 코드는 Apache-2.0 (Moore-AnimateAnyone 기준). 가중치 라이선스는 코드와 별개이며,
  특히 SD1.5는 CreativeML OpenRAIL-M이다. 크레딧: Moore-AnimateAnyone, (원본)
  AnimateAnyone, AnimateDiff, DWPose, Stable Diffusion. 배포 전 각 모델 카드에서
  최종 라이선스 조건을 확인할 것.
