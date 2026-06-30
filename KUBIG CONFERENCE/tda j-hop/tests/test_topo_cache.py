"""위상 캐시(tda.topology.cache) — 적중값이 '새로 계산한 값'과 정확히 일치함을 증명.

캐시는 근사가 아니라 메모이제이션이므로, 입력이 같으면 출력이 같아야 한다(CPU 에선 정확히).
"""
import os
import tempfile

import numpy as np
import torch

from tda.topology.cache import compute_channel_topology_cached, topo_cache_key
from tda.topology.epd import compute_channel_topology

CPU = torch.device("cpu")
PDGNN = {"knn_k": 5, "hks_K": 2, "hop": 1, "max_nodes": 10, "n_train_samples": 6,
         "hidden_dim": 8, "layers": 2, "epochs": 2, "lr": 1e-3, "pi_resolution": 3}


def _toy_adj(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, n, generator=g) > 0.7).float()
    a = ((a + a.t()) > 0).float()
    a.fill_diagonal_(0.0)
    return a


def _cfg():
    return {"pdgnn": dict(PDGNN)}


def test_cache_hit_equals_fresh():
    adj, cfg = _toy_adj(), _cfg()
    with tempfile.TemporaryDirectory() as d:
        # 1) MISS -> 계산+저장 (.npy 1개 생성)
        a1 = compute_channel_topology_cached(adj, cfg, seed=7, device=CPU,
                                             cache_dir=d, tag="toy__gtn__ch0")
        assert len([f for f in os.listdir(d) if f.endswith(".npy")]) == 1
        # 2) HIT(+verify) -> 로드값이 새 계산값과 일치(CPU 결정적이라 max|Δ|=0, verify 통과)
        a2 = compute_channel_topology_cached(adj, cfg, seed=7, device=CPU,
                                             cache_dir=d, verify=True, tol=1e-6,
                                             tag="toy__gtn__ch0")
        assert np.array_equal(a1, a2)                 # 적중값 == 저장 바이트
        # 3) 캐시 미사용 경로(=기존 동작)와도 동일
        a3 = compute_channel_topology(adj, cfg, seed=7, device=CPU)
        assert np.allclose(a1, a3, atol=1e-6)


def test_cache_key_sensitivity():
    adj, cfg = _toy_adj(), _cfg()
    k0 = topo_cache_key(adj, cfg["pdgnn"], 7, "t", "cpu")
    assert topo_cache_key(adj, cfg["pdgnn"], 8, "t", "cpu") != k0          # seed 다름
    assert topo_cache_key(adj, dict(PDGNN, hks_K=3), 7, "t", "cpu") != k0  # 설정 다름
    assert topo_cache_key(_toy_adj(seed=1), cfg["pdgnn"], 7, "t", "cpu") != k0  # adj 다름
    assert topo_cache_key(adj, cfg["pdgnn"], 7, "t", "cpu") == k0          # 동일 입력 → 동일 키


def test_no_cache_dir_passthrough():
    adj, cfg = _toy_adj(), _cfg()
    out = compute_channel_topology_cached(adj, cfg, seed=7, device=CPU, cache_dir=None)
    assert out.shape[0] == adj.size(0)
