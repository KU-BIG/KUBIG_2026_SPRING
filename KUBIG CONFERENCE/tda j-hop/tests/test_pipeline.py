"""파이프라인 통합 테스트 (toy 합성 데이터, CPU). 성능 주장 아님 — 동작/형상 검증."""
import os

import torch

from tda.data import get_dataset, list_datasets
from tda.models.gtn import GTN
from tda.models.fusion import SemanticAttentionFusion
from tda.models.han import HAN
from tda.train import run, _build_A_stack, _build_edge_index_dict
from tda.utils import load_json

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "toy.json")


def _cfg():
    return load_json(os.path.abspath(CFG))


def test_registry_lists_toy_and_acm():
    assert "toy" in list_datasets()
    assert "acm" in list_datasets()


def test_build_bundle_toy_shapes():
    b = get_dataset("toy", _cfg())
    assert b.target == "paper"
    assert b.x.shape[0] == b.num_nodes
    assert set(b.base_relations) == {"PAP", "PSP", "PcP", "PTP"}
    for adj in b.base_relations.values():
        assert adj.shape == (b.num_nodes, b.num_nodes)
        assert torch.all((adj == 0) | (adj == 1))  # 이진
    assert set(b.han_metapaths) == {"PAP", "PSP"}
    for ei in b.han_metapaths.values():
        assert ei.shape[0] == 2


def test_gtn_forward_shapes():
    b = get_dataset("toy", _cfg())
    dev = torch.device("cpu")
    A = _build_A_stack(b, dev)
    assert A.dim() == 3 and A.size(1) == b.num_nodes
    gtn = GTN(num_relations=A.size(0), num_channels=2, in_dim=b.x.size(1),
              hidden_dim=16, num_classes=b.num_classes, num_layers=2)
    logits, H = gtn(A, b.x)
    assert logits.shape == (b.num_nodes, b.num_classes)
    assert H.shape == (2, b.num_nodes, b.num_nodes)


def test_fusion_shape_and_beta():
    fusion = SemanticAttentionFusion(in_dim=10)
    feats = [torch.randn(7, 10), torch.randn(7, 10), torch.randn(7, 10)]
    fused = fusion(feats)
    assert fused.shape == (7, 10)
    assert abs(float(fusion.last_beta.sum()) - 1.0) < 1e-5


def test_han_forward():
    b = get_dataset("toy", _cfg())
    dev = torch.device("cpu")
    eid = _build_edge_index_dict(b, dev)
    han = HAN(in_dim=b.x.size(1), hidden_dim=16, num_classes=b.num_classes,
              target=b.target, metapath_edge_types=list(eid.keys()), heads=2, dropout=0.3)
    out = han(b.x, eid)
    assert out.shape == (b.num_nodes, b.num_classes)


def test_run_baseline_no_topology():
    cfg = _cfg()
    cfg["use_topology"] = False
    rec = run(cfg, "toy", data_root="./data", device=torch.device("cpu"), verbose=False)
    assert 0.0 <= rec["test_macro_f1"] <= 1.0
    assert rec["use_topology"] is False


def test_run_full_pipeline_with_topology(tmp_path):
    cfg = _cfg()
    cfg["use_topology"] = True
    rec = run(cfg, "toy", data_root="./data", device=torch.device("cpu"),
              output_dir=str(tmp_path / "toyrun"), verbose=False)
    assert 0.0 <= rec["test_macro_f1"] <= 1.0
    assert rec["topo_dim"] == 5 * 5 * cfg["pdgnn"]["hks_K"]
    assert len(rec["fusion_beta"]) == cfg["gtn"]["num_channels"]
    assert rec["topology_source"] == "gtn"
    assert rec["gtn_only_test_macro_f1"] is not None  # 실험 A3 기록
    assert (tmp_path / "toyrun" / "metrics.json").exists()


def test_run_random_topology_source_D5():
    cfg = _cfg(); cfg["use_topology"] = True; cfg["topology_source"] = "random"
    rec = run(cfg, "toy", data_root="./data", device=torch.device("cpu"), verbose=False)
    assert rec["topology_source"] == "random"
    assert rec["random_metapath"] is True
    assert 0.0 <= rec["test_macro_f1"] <= 1.0


def test_run_agg_sum_only_D2():
    cfg = _cfg(); cfg["use_topology"] = True; cfg["pdgnn"]["agg"] = "sum"
    rec = run(cfg, "toy", data_root="./data", device=torch.device("cpu"), verbose=False)
    assert 0.0 <= rec["test_macro_f1"] <= 1.0


def test_run_topology_only_and_permutation_diagnostics():
    cfg = _cfg(); cfg["use_topology"] = True
    cfg["node_features"] = "off"; cfg["permute_topology"] = True
    rec = run(cfg, "toy", data_root="./data", device=torch.device("cpu"), verbose=False)
    assert rec["node_features"] == "off"
    assert "test_macro_f1_permuted" in rec
    assert 0.0 <= rec["test_macro_f1_permuted"] <= 1.0


def test_pdgnn_agg_sum_only_shapes():
    from tda.models.pdgnn import PDGNN
    m = PDGNN(hidden_dim=8, num_layers=2, agg="sum")
    filt = torch.randn(5, 1)
    ei = torch.tensor([[0, 1, 2, 1], [1, 2, 3, 0]], dtype=torch.long)
    assert m(filt, ei).shape == (ei.size(1), 2)


def test_run_manual_topology_source_B(tmp_path):
    # 실험 B: GTN 없이 고정 수동 메타패스(PAP/PSP) 위 PDGNN-EPD
    cfg = _cfg()
    cfg["use_topology"] = True
    cfg["topology_source"] = "manual"
    rec = run(cfg, "toy", data_root="./data", device=torch.device("cpu"),
              output_dir=str(tmp_path / "manualrun"), verbose=False)
    assert rec["topology_source"] == "manual"
    assert rec["manual_metapaths"] == cfg["han_metapaths"]
    assert 0.0 <= rec["test_macro_f1"] <= 1.0
    # manual 모드엔 GTN 산출물 없음
    assert "gtn_only_test_macro_f1" not in rec
    assert len(rec["fusion_beta"]) == len(cfg["han_metapaths"])
