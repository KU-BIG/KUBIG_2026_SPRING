"""
Anime Action Scene Generator — Gradio Web UI
실행: python run_gradio.py → http://localhost:7860

파이프라인:
  2D (Moore-AnimateAnyone) : DWPose 2D 스켈레톤 → SD 1.5 or Anything V5
  3D (Champ)               : SMPL 3D 메시 → SD 1.5 or Anything V5  ← 핵심 기여

M5 MacBook(MPS) + RTX 3060(CUDA) 자동 지원.
"""

import subprocess
import sys
import torch
import gradio as gr
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from tools.vid2pose import extract_pose_from_video
from scripts.pose2vid import load_models, build_pipeline, run_inference, get_device, get_weight_dtype

# ─── 파이프라인 설정 테이블 ────────────────────────────────────────────────────
PIPELINES = {
    "2D — SD 1.5 (Baseline)":       {"mode": "2d", "config": "configs/prompts/animation.yaml"},
    "2D — Anything V5 (Anime)":     {"mode": "2d", "config": "configs/prompts/animation_anime.yaml"},
    "3D Champ — SD 1.5":            {"mode": "3d", "config": "configs/champ/inference_3060.yaml"},
    "3D Champ — Anything V5 (Anime)": {"mode": "3d", "config": "configs/champ/inference_3060_anime.yaml"},
}


def apply_speed(video_path: str, speed: float) -> str:
    """ffmpeg로 배속 처리. speed=1.0이면 원본 그대로."""
    if abs(speed - 1.0) < 0.01:
        return video_path
    out_path = str(Path(video_path).with_suffix("")) + f"_x{speed:.1f}.mp4"
    pts = f"setpts={1.0 / speed:.4f}*PTS"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", pts, "-an", out_path],
        check=True, capture_output=True,
    )
    return out_path


class AnimateController:
    def __init__(self):
        self.pipe_2d    = None
        self.pipe_3d    = None
        self.loaded_2d  = None
        self.loaded_3d  = None
        self.device      = get_device()
        self.weight_dtype = get_weight_dtype(self.device)
        print(f"[Device] {self.device} | dtype: {self.weight_dtype}")

    # ─── 2D 파이프라인 로드 ───────────────────────────────────────────────────
    def load_2d(self, config_path: str):
        if self.pipe_2d and self.loaded_2d == config_path:
            return
        print(f"Loading 2D pipeline: {config_path}")
        config = OmegaConf.load(config_path)
        vae, ref_unet, denoise_unet, pose_guider, image_enc, infer_config = load_models(
            config, self.weight_dtype, self.device
        )
        self.pipe_2d   = build_pipeline(vae, image_enc, ref_unet, denoise_unet,
                                        pose_guider, infer_config, self.weight_dtype, self.device)
        self.loaded_2d = config_path
        self._2d_config = config
        print("2D pipeline ready.")

    # ─── 3D 파이프라인 로드 ───────────────────────────────────────────────────
    def load_3d(self, config_path: str):
        if self.pipe_3d and self.loaded_3d == config_path:
            return
        print(f"Loading 3D (Champ) pipeline: {config_path}")
        try:
            from scripts.run_champ import load_champ_pipeline
            cfg = OmegaConf.load(config_path)
            self.pipe_3d   = load_champ_pipeline(cfg, self.device, self.weight_dtype)
            self.loaded_3d = config_path
            self._3d_config = cfg
            print("3D pipeline ready.")
        except ImportError as e:
            raise RuntimeError(
                f"Champ 로드 실패: {e}\n"
                "champ/ 서브모듈이 있는지, 의존성(pip install -r champ/requirements.txt)을 설치했는지 확인하세요."
            )

    # ─── 통합 애니메이션 함수 ─────────────────────────────────────────────────
    def animate(self, ref_image_path, driving_video_path, pipeline_name,
                width, height, n_frames, steps, cfg_scale, seed, speed):

        if ref_image_path is None or driving_video_path is None:
            return None, "입력 파일을 모두 업로드해주세요."

        pipeline_cfg = PIPELINES[pipeline_name]
        mode         = pipeline_cfg["mode"]
        config_path  = pipeline_cfg["config"]

        save_dir = Path("output/gradio") / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator()
        generator.manual_seed(seed)

        # ─── 2D 모드 ──────────────────────────────────────────────────────────
        if mode == "2d":
            self.load_2d(config_path)

            pose_video_path = str(
                Path(driving_video_path).parent
                / (Path(driving_video_path).stem + "_kps.mp4")
            )
            extract_pose_from_video(
                video_path=driving_video_path,
                output_path=pose_video_path,
                device=self.device if self.device != "mps" else "cpu",
            )
            out_path = run_inference(
                pipe=self.pipe_2d,
                ref_image_path=ref_image_path,
                pose_video_path=pose_video_path,
                width=width, height=height,
                n_frames=n_frames, steps=steps, cfg=cfg_scale,
                generator=generator, save_dir=save_dir,
            )

        # ─── 3D 모드 (Champ) ──────────────────────────────────────────────────
        else:
            self.load_3d(config_path)

            # 전처리: 영상 → SMPL 가이드
            guidance_dir = save_dir / "guidance"
            from scripts.preprocess_smpl import extract_frames, run_dwpose, run_hmr2_smpl, render_smpl_guidance
            frames_dir   = guidance_dir / "_frames"
            frame_paths, _ = extract_frames(driving_video_path, frames_dir, max_frames=n_frames)

            run_dwpose(frame_paths, guidance_dir / "dwpose", self.device)

            smpl_params = run_hmr2_smpl(frame_paths, guidance_dir / "_smpl", self.device)
            if smpl_params:
                render_smpl_guidance(smpl_params, frame_paths,
                    out_dirs={k: guidance_dir / k for k in ["depth", "normal", "semantic"]},
                    width=width, height=height)

            # Champ 추론
            from scripts.run_champ import run_champ_inference
            cfg_obj = OmegaConf.load(config_path)
            cfg_obj.ref_image_path  = ref_image_path
            cfg_obj.guidance_folder = str(guidance_dir)
            cfg_obj.output_dir      = str(save_dir)
            cfg_obj.width           = width
            cfg_obj.height          = height
            cfg_obj.num_frames      = n_frames
            cfg_obj.num_inference_steps = steps
            cfg_obj.guidance_scale  = cfg_scale
            cfg_obj.seed            = seed
            out_path = run_champ_inference(self.pipe_3d, cfg_obj, self.device, seed=seed)

        # ─── 배속 후처리 ──────────────────────────────────────────────────────
        try:
            out_path = apply_speed(out_path, speed)
        except Exception as e:
            print(f"[WARN] 배속 처리 실패: {e}")

        return out_path, f"완료 ({pipeline_name}): {out_path}"


controller = AnimateController()

# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Anime Action Scene Generator") as demo:
    gr.Markdown("## Anime Action Scene Generator")
    gr.Markdown(
        "캐릭터 이미지 + 액션 영상 → 애니 스타일 액션씬 생성  \n"
        f"**실행 환경**: `{controller.device}` | `{controller.weight_dtype}`"
    )

    with gr.Row():
        with gr.Column():
            ref_image    = gr.Image(type="filepath", label="캐릭터 이미지 (Reference)")
            drive_video  = gr.Video(label="액션 영상 (Driving Video)")

            pipeline_choice = gr.Radio(
                choices=list(PIPELINES.keys()),
                value="3D Champ — Anything V5 (Anime)",
                label="파이프라인 선택",
            )
            gr.Markdown(
                "_2D: DWPose(2D 스켈레톤) 기반 · "
                "3D Champ: SMPL(3D 신체 메시) 기반 — 액션씬에 더 안정적_"
            )

            with gr.Accordion("생성 설정", open=True):
                speed     = gr.Slider(0.5, 3.0, value=1.5, step=0.25, label="배속")
                cfg_scale = gr.Slider(1.0, 10.0, value=3.5, step=0.5,  label="CFG Scale")
                n_frames  = gr.Slider(8, 32, value=16, step=4, label="생성 프레임 수")

            with gr.Accordion("고급 설정", open=False):
                width  = gr.Slider(256, 512, value=384, step=64, label="Width")
                height = gr.Slider(384, 768, value=512, step=64, label="Height")
                steps  = gr.Slider(10, 50,  value=20,  step=5,  label="DDIM Steps")
                seed   = gr.Number(value=42, label="Seed", precision=0)

            run_btn = gr.Button("Generate Action Scene", variant="primary", size="lg")

        with gr.Column():
            output_video = gr.Video(label="결과 영상")
            status_text  = gr.Textbox(label="상태", interactive=False)

    run_btn.click(
        fn=controller.animate,
        inputs=[ref_image, drive_video, pipeline_choice,
                width, height, n_frames, steps, cfg_scale, seed, speed],
        outputs=[output_video, status_text],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
