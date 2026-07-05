"""LP(Level 1) 단위·통합 테스트 — 분할 누출·mix 불변량·end-to-end (toy, CPU)."""
import copy
import json

import numpy as np
import torch

from tda.lp import auc_score, degree_mix, run_lp, sample_negatives, split_edges

CPU = torch.device("cpu")


def _lp_config():
    cfg = json.load(open("configs/toy.json"))
    tgt = list(cfg["base_relations"].keys())[0]
    cfg["lp"] = {"target_relation": tgt, "hidden": 16, "num_layers": 2, "dropout": 0.0,
                 "lr": 0.01, "weight_decay": 5e-4, "epochs": 15, "edge_cap": 200,
                 "split_seed": 123, "topo_seed": 777, "topo_cache": None}
    # 테스트에선 캐시 미사용(None → 원본 호출) + 작은 pdgnn
    cfg["pdgnn"].update({"n_train_samples": 6, "epochs": 2, "hidden_dim": 8, "layers": 2})
    return cfg


def test_split_deterministic_and_no_overlap():
    A = (torch.rand(40, 40) > 0.7).float()
    A = ((A + A.t()) > 0).float()
    A.fill_diagonal_(0)
    tr1, va1, te1, pos = split_edges(A, cap=0, split_seed=7)
    tr2, va2, te2, _ = split_edges(A, cap=0, split_seed=7)
    assert torch.equal(te1, te2) and torch.equal(va1, va2)      # 고정 분할
    sets = [set(map(tuple, t.tolist())) for t in (tr1, va1, te1)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    negs = sample_negatives(30, 40, pos, np.random.RandomState(0))
    assert not (set(map(tuple, negs.tolist())) & pos)           # 음성 ∩ 양성 = ∅


def test_degree_mix_preserves_multiset_within_bucket():
    n = 50
    topo = [torch.randn(n, 8)]
    deg = torch.randint(0, 10, (n,)).float()
    mixed, perm = degree_mix(topo, deg, seed=1, n_bucket=5)
    # 순열이므로 전체 multiset 보존 + 실제로 섞임
    assert torch.allclose(mixed[0].sort(0).values, topo[0].sort(0).values)
    assert not torch.equal(perm, torch.arange(n))


def test_auc_sanity():
    assert auc_score(torch.tensor([2.0, 3.0]), torch.tensor([0.0, 1.0])) == 1.0
    assert abs(auc_score(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) - 0.5) < 1e-6


def test_run_lp_end_to_end_conditions():
    base = _lp_config()
    aucs = {}
    for name, ov in [("a", {"use_topology": False}),
                     ("b1", {"use_topology": True, "topology_noise": "random"}),
                     ("c", {"use_topology": True}),
                     ("m", {"use_topology": True, "topology_mode": "degree_mix"})]:
        cfg = copy.deepcopy(base)
        cfg.update(ov)
        cfg["seed"] = 0
        r = run_lp(cfg, "toy", "./data", device=CPU, verbose=False)
        assert 0.0 <= r["test_auc"] <= 1.0 and np.isfinite(r["test_auc"])
        aucs[name] = r["test_auc"]
    assert aucs["a"] > 0.55    # toy 는 구조가 강해 기준 encoder 만으로 chance 초과
