"""
2d/noise.py
Moore-AnimateAnyone의 초기 latent noise 방식을 바꾸기 위한 파일

Pose2VideoPipeline은 기본적으로 prepare_latents() 안에서
프레임별 IID noise를 생성

이 파일은 pipeline의 prepare_latents를 monkeypatch로 교체하여,
학습 없이 초기 noise 방식만 iid / repeat / freenoise 중 하나로 바꿈.

모든 전략은 latent shape (B, C, F, H/8, W/8)에서
frame 축 F를 어떻게 채울지만 다르게 만든다.
"""

import types
import torch


try:
    from diffusers.utils.torch_utils import randn_tensor
except Exception:  # pragma: no cover - only hit when diffusers is absent (tests)
    def randn_tensor(shape, generator=None, device=None, dtype=None):
        device = torch.device(device) if device is not None else torch.device("cpu")
        if generator is not None and generator.device != device:
            # generator의 device에서 먼저 샘플링한 후 이동 (diffusers의 reproducibility 처리 방식)
            t = torch.randn(shape, generator=generator, device=generator.device, dtype=dtype)
            return t.to(device)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)


STRATEGIES = ("iid", "repeat", "freenoise")

DEFAULT_WINDOW_SIZE = 24    # pipeline의 context_frames와 맞춤
DEFAULT_WINDOW_STRIDE = 4   # pipeline의 context_overlap과 맞춤


def _randperm(n, generator, device):
    """generator의 device와 무관하게 재현 가능한 range(n) 순열을 device 위에 생성한다
    (torch RNG 연산은 device가 서로 일치해야 하기 때문)."""
    if generator is not None:
        return torch.randperm(n, generator=generator, device=generator.device).to(device)
    return torch.randperm(n, device=device)


def iid_noise(shape, generator, device, dtype):
    """프레임마다 독립적인 Gaussian noise 생성 (upstream 기본값)"""
    return randn_tensor(shape, generator=generator, device=device, dtype=dtype)


def repeat_noise(shape, generator, device, dtype):
    """한 프레임의 noise를 샘플링해서 모든 프레임에 broadcast"""
    b, c, f, h, w = shape
    one = randn_tensor((b, c, 1, h, w), generator=generator, device=device, dtype=dtype)
    return one.repeat(1, 1, f, 1, 1)


def freenoise(shape, generator, device, dtype,
              window_size=DEFAULT_WINDOW_SIZE, window_stride=DEFAULT_WINDOW_STRIDE):
    """FreeNoise noise rescheduling (tuning 없이도 긴 영상에서 일관성 유지)

    window_size 길이의 base window를 iid로 샘플링한 후, 나머지 프레임은 이전
    window_stride 길이 블록(window_size만큼 앞선 위치)을 블록 단위로 local shuffle해서
    복사해 채운다. 이렇게 하면 멀리 떨어진 프레임끼리도 상관관계를 유지하면서, shuffle
    덕분에 완전히 반복되는 정적인 느낌은 피할 수 있다. 참고: Qiu et al., "FreeNoise".
    """
    b, c, f, h, w = shape
    if window_size < 1 or window_stride < 1:
        raise ValueError("window_size and window_stride must be >= 1")
    if window_stride > window_size:
        raise ValueError(
            f"window_stride ({window_stride}) must be <= window_size ({window_size})"
        )

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    if f <= window_size:
        return latents  # 재배치하기엔 프레임이 짧음 -> iid와 동일

    for frame_index in range(window_size, f, window_stride):
        n = min(window_stride, f - frame_index)                 # 마지막 블록은 더 짧을 수 있음
        start = frame_index - window_size                       # 항상 >= 0
        src = torch.arange(start, start + window_stride, device=device)
        src = src[_randperm(window_stride, generator, device)][:n]
        # src가 destination 범위보다 항상 앞이므로 in-place copy를 해도 안전하다
        latents[:, :, frame_index:frame_index + n] = latents[:, :, src]
    return latents


def make_prepare_latents(strategy, window_size=DEFAULT_WINDOW_SIZE,
                         window_stride=DEFAULT_WINDOW_STRIDE):
    """주어진 전략에 대한 prepare_latents 교체용 bound-method를 생성한다.

    시그니처는 Pose2VideoPipeline.prepare_latents와 완전히 동일하다 (주의: pipeline의
    positional call 순서에 맞춰 width가 height보다 먼저 온다). 반환되는 latents는
    * scheduler.init_noise_sigma 스케일링을 그대로 유지한다.
    """
    strategy = str(strategy).lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown noise strategy '{strategy}'; choose from {STRATEGIES}")

    def prepare_latents(self, batch_size, num_channels_latents, width, height,
                        video_length, dtype, device, generator, latents=None):
        # 명시적으로 전달된 latent가 있으면 원본과 동일하게 그대로 사용한다.
        if latents is not None:
            return latents.to(device) * self.scheduler.init_noise_sigma

        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if strategy == "repeat":
            latents = repeat_noise(shape, generator, device, dtype)
        elif strategy == "freenoise":
            latents = freenoise(shape, generator, device, dtype, window_size, window_stride)
        else:  # iid
            latents = iid_noise(shape, generator, device, dtype)

        return latents * self.scheduler.init_noise_sigma

    return prepare_latents


def install(pipe, strategy, window_size=DEFAULT_WINDOW_SIZE,
            window_stride=DEFAULT_WINDOW_STRIDE):
    """선택한 전략으로 pipe.prepare_latents를 monkeypatch하고 pipe를 반환한다"""
    pipe.prepare_latents = types.MethodType(
        make_prepare_latents(strategy, window_size, window_stride), pipe
    )
    return pipe
