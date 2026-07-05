"""LP Level 2 (pair-vicinity EPD) 테스트 — vicinity·캐시·CN-mix·end-to-end (toy, CPU)."""
import copy
import json
import tempfile

import numpy as np
import torch

from tda.lp import auc_score, cn_bucket_mix, cn_counts, run_lp
from tda.topology.pair_epd import _pair_vicinity, compute_pair_topology_cached, nx_from_dense

CPU = torch.device("cpu")


def _cfg(pair_feature):
    cfg = json.load(open("configs/toy.json"))
    tgt = list(cfg["base_relations"].keys())[0]
    cfg["lp"] = {"target_relation": tgt, "hidden": 16, "num_layers": 2, "dropout": 0.0,
                 "lr": 0.01, "weight_decay": 5e-4, "epochs": 12, "edge_cap": 150,
                 "split_seed": 123, "topo_seed": 777, "topo_cache": None,
                 "pair_feature": pair_feature, "fixed_train_neg": True, "pair_cache": None}
    cfg["use_topology"] = False
    cfg["pdgnn"].update({"n_train_samples": 5, "epochs": 2, "hidden_dim": 8, "layers": 2})
    return cfg


def _toy_adj(n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, n, generator=g) > 0.75).float()
    a = ((a + a.t()) > 0).float()
    a.fill_diagonal_(0)
    return a


def test_pair_vicinity_contains_endpoints_and_caps():
    adj = _toy_adj()
    g = nx_from_dense(adj)
    nf = {nd: float(nd) for nd in g.nodes()}
    filt, ei = _pair_vicinity(g, 0, 5, hop=1, node_filt=nf, max_nodes=8)
    assert len(filt) <= 10          # cap(+u,v 보존 여유)
    assert ei.shape[0] == 2


def test_pair_topology_cache_roundtrip():
    adj = _toy_adj()
    pairs = np.array([[0, 5], [1, 7], [2, 9]], dtype=np.int64)
    cfg = _cfg("real")
    with tempfile.TemporaryDirectory() as d:
        f1 = compute_pair_topology_cached(adj, pairs, cfg, 777, d, "t", device=CPU)
        f2 = compute_pair_topology_cached(adj, pairs, cfg, 777, d, "t", device=CPU)
        assert np.array_equal(f1, f2)                # 캐시 적중 = 동일
        assert f1.shape == (3, cfg["pdgnn"]["pi_resolution"] ** 2 * cfg["pdgnn"]["hks_K"])


def test_cn_bucket_mix_preserves_multiset():
    A = _toy_adj()
    pairs = torch.randint(0, 30, (40, 2))
    feats = torch.randn(40, 6)
    cn = cn_counts(A, pairs)
    mixed = cn_bucket_mix(feats, cn, seed=0)
    assert torch.allclose(mixed.sort(0).values, feats.sort(0).values)


def test_run_lp2_end_to_end():
    aucs = {}
    for pk in ["none", "real", "noise", "mix"]:
        cfg = _cfg(pk)
        cfg["seed"] = 0
        r = run_lp(cfg, "toy", "./data", device=CPU, verbose=False)
        assert np.isfinite(r["test_auc"]) and 0 <= r["test_auc"] <= 1
        if pk != "none":
            assert "test_auc_cn" in r               # CN baseline 기록
        aucs[pk] = r["test_auc"]
    assert aucs["none"] > 0.55