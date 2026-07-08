# 2d/ — UniAnimate 2D Inference Wrapper

UniAnimate submodule 기반 2D character animation inference wrapper.
공개 가중치를 사용해 reference image + driving video로 animation video를 생성한다.

## 폴더 구조

```text
2d/
├── UniAnimate/                 # original UniAnimate submodule
├── README.md
├── requirements.txt
└── UniAnimate_infer_my.yaml
```

## Setup

먼저 submodule을 받아온 뒤, 원본 UniAnimate 폴더 안에서 실행 환경을 설치한다.

```bash
git submodule add https://github.com/ali-vilab/UniAnimate.git UniAnimate
git submodule update --init --recursive

cd UniAnimate
pip install -r ../requirements.txt
```

## Weights

UniAnimate 공개 가중치를 다운로드한 뒤, 코드에서 바로 사용할 수 있는 위치로 이동한다.

```bash
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('iic/unianimate', cache_dir='checkpoints/')
PY

mv checkpoints/iic/unianimate/* checkpoints/
```

## Run

reference image와 driving video를 준비하고 driving video에서 pose를 추출한다.

```bash
mkdir -p data/images data/videos data/saved_pose
cp /workspace/ref.jpg data/images/ref.jpg
# driving video: data/videos/source_clip.mp4


python run_align_pose.py   --ref_name data/images/ref.jpg   --source_video_paths data/videos/source_clip.mp4   --saved_pose_dir data/saved_pose/ref_clip

cp ../UniAnimate_infer_my.yaml configs/UniAnimate_infer_my.yaml
python inference.py --cfg configs/UniAnimate_infer_my.yaml
```
