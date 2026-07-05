"""실험 ①(Gated-RGCN factorial) 테스트 — 게이트 경로·조건 end-to-end (toy, CPU)."""
import copy
import json

import numpy as np
import torch

from tda.gated import run_gated
from tda.models.gated_rgcn import GatedRGCN

CPU = torch.device("cpu")


def _cfg():
    cfg = json.load(open("configs/toy.json"))
    cfg["gated"] = {"hidden": 16, "num_layers": 2, "dropout": 0.0, "lr": 0.01,
                    "weight_decay": 5e-4, "epochs": 15, "gate_hidden": 16,
                    "topo_seed": 777, "topo_cache": None}
    cfg["pdgnn"].update({"n_train_samples": 6, "epochs": 2, "hidden_dim": 8, "layers": 2})
    return cfg


def test_gate_off_equals_plain():
    """gate_feat 미제공 시 GatedRGCN == 게이트 없는 동일 파라미터 forward."""
    torch.manual_seed(0)
    n, r = 12, 2
    x = torch.randn(n, 5)
    ei = torch.randint(0, n, (2, 30))
    et = torch.randint(0, r, (30,))
    m = GatedRGCN(5, 8, 3, r, gate_dim=4)
    m.eval()
    out1 = m(x, ei, et)                       # gate_feat 없음 → 게이트 미사용
    out2 = m(x, ei, et, gate_feat=None)
    assert torch.allclose(out1, out2)
    # gate_feat 제공 시엔 값이 달라져야(게이트 실제 작동)
    out3 = m(x, ei, et, gate_feat=torch.randn(n, 4))
    assert not torch.allclose(out1, out3)


def test_seed_diversity_survives_topology_cache(tmp_path):
    """회귀: 캐시 적중이 전역 RNG 를 복원해도 run seed 재시드로 seed 간 결과가 달라야 함.
    (버그: *_real 조건 std=0.000 — 10 seed 가 전부 동일 run 이 됐던 사고)"""
    base = _cfg()
    base["gated"].update(injection="concat", content="real",
                         topo_cache=str(tmp_path / "cache"))
    outs = []
    for s in (0, 1):
        cfg = copy.deepcopy(base)
        cfg["seed"] = s
        outs.append(run_gated(cfg, "toy", "./data", device=CPU, verbose=False)["test_macro_f1"])
    # seed 0 이 캐시를 쓰고 seed 1 이 적중 → 그래도 결과는 달라야 함
    assert outs[0] != outs[1], f"seed 0/1 결과 동일({outs[0]}) — RNG 복원 오염 재발"


def test_run_gated_all_conditions():
    base = _cfg()
    res = {}
    for suf, inj, cont in [("base", "none", "real"),
                           ("cat_real", "concat", "real"), ("cat_noise", "concat", "noise"),
                           ("cat_mix", "concat", "mix"),
                           ("gate_real", "gate", "real"), ("gate_noise", "gate", "noise"),
                           ("gate_mix", "gate", "mix")]:
        cfg = copy.deepcopy(base)
        cfg["gated"].update(injection=inj, content=cont)
        cfg["seed"] = 0
        r = run_gated(cfg, "toy", "./data", device=CPU, verbose=False)
        assert np.isfinite(r["test_macro_f1"]) and 0 <= r["test_macro_f1"] <= 1
        res[suf] = r["test_macro_f1"]
    assert res["base"] > 0.5      # toy 는 base 만으로 학습 가능
