"""Stage 2 위상특징(PDGNN persistence image) 디스크 캐시 — *재실험 전용, 기본 OFF*.

`compute_channel_topology(adj, config, seed)` 의 결과 (N, res^2*K) 를 **입력의 정확한 해시**
로 키잉해 저장/재사용한다.

  - **적중(hit) = 입력 바이트가 완전히 동일 = 출력 동일** 이므로 수치/재현성에 영향이 없다
    (근사가 아니라 단순 메모이제이션).
  - 키가 조금이라도 다르면(다른 채널 adj/다른 pdgnn 설정/다른 seed) **miss → 새로 계산**.
    즉 잘못 재사용으로 결과가 오염될 일이 없다. 손해 봐야 "속도뿐"이고 값은 항상 올바르다.

쓰임: 위상(stage2)은 다운스트림 분류기(HAN/fusion/lr/epoch)와 **독립**이므로, 한 번 계산해
두면 다운스트림만 바꾸는 재실험에서 ~17분짜리 stage2 를 통째로 건너뛸 수 있다.

주의(충실도): GPU 커널 비결정성 때문에 동일 입력이라도 실행마다 PI 가 ~1e-6 수준 다를 수
있다. 캐시는 '처음 계산된 값'을 고정 반환하므로 적중 시 위상이 캐시로 **결정화**된다(재현성↑).
`verify=True` 면 적중 시 새로 계산해 최대 절대오차를 출력/검증한다(CPU 에선 정확히 0).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from tda.topology.epd import compute_channel_topology


TOPOLOGY_CACHE_SCHEMA_VERSION = "gtn_pdgnn_topology_v3_device_independent"


def _rng_metadata(device) -> dict:
    meta = {"torch_rng_state": torch.get_rng_state().cpu().tolist()}
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if getattr(dev, "type", str(dev)) == "cuda" and torch.cuda.is_available():
        meta["cuda_rng_state"] = torch.cuda.get_rng_state(dev).cpu().tolist()
    return meta


def _restore_rng_metadata(meta: dict, device) -> bool:
    state = meta.get("torch_rng_state")
    if state is None:
        return False
    torch.set_rng_state(torch.tensor(state, dtype=torch.uint8))
    cuda_state = meta.get("cuda_rng_state")
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if getattr(dev, "type", str(dev)) == "cuda" and torch.cuda.is_available():
        if cuda_state is None:
            return False
        torch.cuda.set_rng_state(torch.tensor(cuda_state, dtype=torch.uint8), device=dev)
    return True


def topo_cache_key(adj: torch.Tensor, pdgnn_cfg: dict, seed: int, tag: str) -> str:
    """위상 출력에 영향을 주는 실험 정의의 SHA1. device/backbone 은 포함하지 않는다."""
    a = np.ascontiguousarray(adj.detach().cpu().numpy())
    h = hashlib.sha1()
    h.update(TOPOLOGY_CACHE_SCHEMA_VERSION.encode())
    h.update(str(tag).encode())
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    h.update(json.dumps(pdgnn_cfg, sort_keys=True, default=str).encode())
    h.update(str(int(seed)).encode())
    return h.hexdigest()


def compute_channel_topology_cached(adj: torch.Tensor, config: dict, seed: int,
                                    device=None, verbose: bool = False,
                                    cache_dir: str = None, verify: bool = False,
                                    tag: str = "", tol: float = 1e-3) -> np.ndarray:
    """`compute_channel_topology` 의 캐시 래퍼.

    cache_dir 가 None/빈값이면 캐시 없이 원본을 그대로 호출(= 기존 동작, 부작용 0).
    """
    if not cache_dir:
        return compute_channel_topology(adj, config, seed, device=device, verbose=verbose)

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    device_type = getattr(dev, "type", str(dev))
    os.makedirs(cache_dir, exist_ok=True)
    key = topo_cache_key(adj, config.get("pdgnn", {}), seed, tag)
    path = os.path.join(cache_dir, key + ".npy")

    if os.path.exists(path):
        arr = np.load(path)
        meta_path = os.path.join(cache_dir, key + ".json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if verify:
            fresh = compute_channel_topology(adj, config, seed, device=device, verbose=verbose)
            if fresh.shape != arr.shape:
                raise ValueError(f"[topology_cache] shape mismatch cached={arr.shape} "
                                 f"fresh={fresh.shape} (key {key[:12]})")
            md = float(np.max(np.abs(fresh - arr))) if arr.size else 0.0
            restored = _restore_rng_metadata(meta, dev)
            suffix = "" if restored else " rng_state=missing"
            print(f"[topology_cache] hit+verify key={key[:12]} max|Δ|={md:.2e} (tol={tol}){suffix}", flush=True)
            if md > tol:
                raise ValueError(f"[topology_cache] verify FAILED max|Δ|={md:.2e} > tol {tol} "
                                 f"(key {key[:12]})")
        else:
            restored = _restore_rng_metadata(meta, dev)
            if verbose:
                suffix = "" if restored else " rng_state=missing"
                print(f"[topology_cache] hit key={key[:12]} path={path}{suffix}", flush=True)
        return arr

    arr = np.asarray(compute_channel_topology(adj, config, seed, device=device, verbose=verbose))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:          # np.save(파일객체) 는 확장자를 붙이지 않음
        np.save(f, arr)
    os.replace(tmp, path)               # 원자적 교체(부분 기록 방지)
    meta = {"schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
            "tag": tag, "seed": int(seed), "shape": list(arr.shape),
            "dtype": str(arr.dtype), "computed_device": device_type,
            "pdgnn": config.get("pdgnn", {})}
    meta.update(_rng_metadata(dev))
    json.dump(meta, open(os.path.join(cache_dir, key + ".json"), "w"), indent=2, default=str)
    if verbose:
        print(f"[topology_cache] miss key={key[:12]} saved={path} shape={arr.shape}", flush=True)
    return arr
