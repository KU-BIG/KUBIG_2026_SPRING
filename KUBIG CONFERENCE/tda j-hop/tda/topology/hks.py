"""Heat Kernel Signature (HKS) 노드 필터.

TLC-GNN `diffusion_features.compute_hks_features` 를 충실히 재구현(의존성 경량화):

    HKS_t(i) = sum_k exp(-t * lambda_k) * phi_k(i)^2

정규화 라플라시안 L = I - D^{-1/2} A D^{-1/2} 의 고유분해(torch.linalg.eigh)를 쓰고,
확산시간 t 를 양의 스펙트럼에 걸쳐 K 개 log-spaced 로 잡는다(국소->전역).
각 스케일 열은 노드 간 z-정규화(mean 0, std 1)한다.

PDGNN 의 노드 스칼라 필터로 이 (N, K) 행렬의 각 열을 사용한다.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def compute_hks(
    edge_index: torch.Tensor,
    num_nodes: int,
    K: int = 3,
    device=None,
    verbose: bool = False,
) -> np.ndarray:
    """(N, K) 다중 스케일 HKS 행렬을 반환.

    edge_index: (2, E) long. 무방향으로 대칭화하여 사용.
    """
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    n = int(num_nodes)
    ei = np.asarray(edge_index.cpu())

    A = torch.zeros((n, n), dtype=torch.float64, device=dev)
    if ei.shape[1] > 0:
        src = torch.from_numpy(ei[0]).long().to(dev)
        dst = torch.from_numpy(ei[1]).long().to(dev)
        A[src, dst] = 1.0
        A[dst, src] = 1.0
    A.fill_diagonal_(0.0)

    deg = A.sum(dim=1)
    dinv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    Dinv = torch.diag(dinv)
    L = torch.eye(n, dtype=torch.float64, device=dev) - Dinv @ A @ Dinv

    t0 = time.time()
    lams, phis = torch.linalg.eigh(L)  # 오름차순 고유값, 대칭
    lams = torch.clamp(lams, min=0.0)
    if verbose:
        print(f"    [hks] eigh n={n} {time.time()-t0:.2f}s "
              f"lambda[{float(lams.min()):.4f},{float(lams.max()):.4f}]", flush=True)

    pos = lams[lams > 1e-8]
    if pos.numel() == 0:
        return np.zeros((n, K), dtype=np.float64)
    lam_min, lam_max = float(pos.min()), float(pos.max())
    # t in [1/lam_max (local), 1/lam_min (global)], log-spaced.
    ts = torch.logspace(
        np.log10(1.0 / lam_max), np.log10(1.0 / lam_min), K, dtype=torch.float64, device=dev
    )

    phis2 = phis * phis  # (N, N) phi_k(i)^2
    hks = torch.zeros((n, K), dtype=torch.float64, device=dev)
    for j, t in enumerate(ts):
        coef = torch.exp(-t * lams)  # (N,)
        hks[:, j] = (phis2 * coef.unsqueeze(0)).sum(dim=1)

    # 스케일별 z-정규화
    mean = hks.mean(dim=0, keepdim=True)
    std = hks.std(dim=0, keepdim=True)
    hks = (hks - mean) / (std + 1e-8)
    return hks.cpu().numpy()
