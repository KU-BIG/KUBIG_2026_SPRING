"""
3d/preprocess.py
raw 영상 -> Champ guidance(depth/normal/semantic_map/dwpose + mask) 생성 오케스트레이터

주의:
    이 경로는 "무거운 별도 환경"이 필요하고, 이 저장소에서는 실행 검증되지 않았다.
    Champ 공식 docs/data_process.md 의 6단계를 그대로 순서대로 호출할 뿐이며, 첫 실전
    실행 시 경로/파일명 규칙을 상황에 맞게 손봐야 할 수 있다. 사전 렌더된 guidance 폴더가
    있으면 이 파일을 쓸 필요 없이 run.py 에 그 폴더를 --video 로 바로 주면 된다.

필요한 것 (생성용 requirements 와 별개):
    - ffmpeg                          영상 -> 프레임 분해
    - 4D-Humans (hmr2) + detectron2   SMPL 추정
    - Blender 3.6                     SMPL 스무딩 / 조건맵 렌더 (pip 아님)
    - SMPL body model (.pkl)          라이선스 필요, smpl.is.tue.mpg.de 에서 수동 다운로드
    - DWPose 리포                     champ/DWPose 에 clone (generate_dwpose 가 참조)
    - HMR2/detectron2 체크포인트       python -m scripts.pretrained_models.download --all

파이프라인 (Champ 원본 명령 그대로):
    1. ffmpeg 로 영상 -> driving_videos/<name>/images/*.png, ref -> reference_imgs/images/*
    2. generate_smpls        (SMPL 피팅)
    3. smooth_smpls          (Blender)
    4. smpl_transfer         (ref 로 리타겟)
    5. render_condition_maps (Blender, depth/normal/semantic_map + mask 렌더)
    6. generate_dwpose       (dwpose 렌더)
    -> transferd_result/ 이 곧 guidance 폴더

실행 방법:
    # run.py 가 --video 에 영상 파일을 받으면 자동으로 build_guidance 를 호출한다.
    # 단독 실행도 가능:
    python preprocess.py --video dance.mp4 --ref character.png --device 0
"""

import argparse
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMODULE = HERE / "champ"   # scripts.data_processors.* 는 이 폴더가 CWD 여야 import 된다

# 전처리 결과물이 쌓이는 위치 (Champ 문서 기준 상대경로, CWD=champ)
DRIVING_DIR = "driving_videos"
REFERENCE_DIR = "reference_imgs"
TRANSFER_DIR = "transferd_result"

BLEND_FILE = "scripts/data_processors/smpl/blend/smpl_rendering.blend"
GUIDANCE_TYPES = ("depth", "normal", "semantic_map", "dwpose")


def _run(cmd, cwd):
    """subprocess 실행 (실패 시 예외). 명령을 먼저 출력한다."""
    print(f"[preprocess] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def check_prereqs():
    """전처리 사전 요건을 확인하고, 없으면 안내와 함께 예외를 던진다."""
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (영상 분해)")
    if shutil.which("blender") is None:
        missing.append("blender 3.6 (스무딩/렌더, PATH 에 있어야 함)")
    if find_spec("hmr2") is None:
        missing.append("4D-Humans(hmr2) — pip install git+https://github.com/shubham-goel/4D-Humans")
    if not (SUBMODULE / "DWPose").is_dir():
        missing.append("DWPose 리포 — git clone https://github.com/IDEA-Research/DWPose champ/DWPose")
    if missing:
        raise RuntimeError(
            "raw 영상 전처리에 필요한 요건이 없습니다:\n  - "
            + "\n  - ".join(missing)
            + "\n\n자세한 설치는 champ/docs/data_process.md 를 참고하세요. "
            "사전 렌더된 guidance 폴더가 있으면 그 폴더를 --video 로 바로 주는 편이 낫습니다."
        )


def build_guidance(video_path, ref_path, frame_range=None, device=0, workdir=None):
    """raw 영상 + ref 이미지 -> guidance 폴더 경로를 반환한다.

    frame_range 는 여기서 쓰지 않는다 (전처리는 전 프레임을 만들고, run.py 의
    load_guidance_group 이 로드 시점에 [min:max] 로 잘라낸다).
    """
    check_prereqs()

    cwd = Path(workdir) if workdir else SUBMODULE
    video_path = os.path.abspath(video_path)
    ref_path = os.path.abspath(ref_path)
    video_name = Path(video_path).stem
    ref_name = Path(ref_path).stem

    # 1) ffmpeg: 영상 -> 프레임, ref -> reference_imgs/images
    frames_dir = cwd / DRIVING_DIR / video_name / "images"
    ref_images_dir = cwd / REFERENCE_DIR / "images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    ref_images_dir.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-i", video_path, "-c:v", "png", str(frames_dir / "%04d.png")], cwd)
    shutil.copy(ref_path, ref_images_dir / f"{ref_name}.png")

    driving_path = f"{DRIVING_DIR}/{video_name}"

    # 2) SMPL 피팅 (ref + driving)
    _run([sys.executable, "-m", "scripts.data_processors.smpl.generate_smpls",
          "--reference_imgs_folder", REFERENCE_DIR,
          "--driving_video_path", driving_path,
          "--device", str(device)], cwd)

    # 3) SMPL 스무딩 (Blender)
    smpls_npz = f"{driving_path}/smpl_results/smpls_group.npz"
    _run(["blender", "--background", "--python",
          "scripts/data_processors/smpl/smooth_smpls.py",
          "--smpls_group_path", smpls_npz,
          "--smoothed_result_path", smpls_npz], cwd)

    # 4) SMPL 리타겟 (ref 의 체형/카메라로)
    _run([sys.executable, "-m", "scripts.data_processors.smpl.smpl_transfer",
          "--reference_path", f"{REFERENCE_DIR}/smpl_results/{ref_name}.npy",
          "--driving_path", driving_path,
          "--output_folder", TRANSFER_DIR,
          "--figure_transfer", "--view_transfer"], cwd)

    # 5) 조건맵 렌더 (Blender): depth / normal / semantic_map + mask
    _run(["blender", BLEND_FILE, "--background", "--python",
          "scripts/data_processors/smpl/render_condition_maps.py",
          "--driving_path", f"{TRANSFER_DIR}/smpl_results",
          "--reference_path", f"{REFERENCE_DIR}/images/{ref_name}.png",
          "--device", str(device)], cwd)

    # 6) DWPose 렌더 (normal 을 입력으로)
    _run([sys.executable, "-m", "scripts.data_processors.dwpose.generate_dwpose",
          "--input", f"{TRANSFER_DIR}/normal",
          "--output", f"{TRANSFER_DIR}/dwpose"], cwd)

    guidance_dir = (cwd / TRANSFER_DIR).resolve()
    missing = [g for g in GUIDANCE_TYPES if not (guidance_dir / g).is_dir()]
    if missing:
        raise RuntimeError(
            f"전처리 후에도 guidance 하위 폴더가 없습니다: {missing} (경로: {guidance_dir}). "
            "각 단계 로그를 확인하고 docs/data_process.md 와 대조하세요."
        )
    print(f"[preprocess] guidance 준비 완료 -> {guidance_dir}")
    return str(guidance_dir)


def parse_args():
    ap = argparse.ArgumentParser(description="raw 영상 -> Champ guidance 전처리")
    ap.add_argument("--video", required=True, help="raw 영상 파일")
    ap.add_argument("--ref", required=True, help="reference 이미지")
    ap.add_argument("--device", type=int, default=0, help="GPU device id")
    ap.add_argument("--workdir", default=None, help="작업 폴더 (기본: champ/)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_guidance(args.video, args.ref, device=args.device, workdir=args.workdir)
    print(out)
