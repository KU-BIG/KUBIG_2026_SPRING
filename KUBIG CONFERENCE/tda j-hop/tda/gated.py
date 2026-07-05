"""실험 ① Gated-RGCN factorial — 주입{none, concat, gate} × 내용{real, noise, mix} (NC).

질문: "(f)<(d) 의 위상 유해가 concat *주입 방식* 탓인가?" 를 격리.
  - 위상: manual 채널(전 base relation) + **topo_seed 고정** + 캐시 → 데이터셋당 1회 계산.
    (f 와 달리 GTN/per-seed 위상이 아님 — 주입 효과를 순수 격리하기 위한 의도적 고정)
  - 모든 조건이 같은 GatedRGCN 구현 사용(base/concat 은 게이트 미사용) → 구현 차이 제거.
  - mix 는 NC 의 class-mix 와 동일한 class·split 내 셔플(tda.train 재사용).
  - 평가: NC 와 동일 (train/val/test mask, best-val macro-F1 선택, test macro-F1/accuracy).

사용:  python -m tda.gated --config configs/campaign/acm__gt_gate_real.json --dataset acm \
          --data-root <root> --output-dir runs/gated/... --seed 0
"""
from __future__ import annotations

import argparse
import copy
import os
from typing import Dict

import torch

from tda.data import get_dataset
from tda.models.fusion import SemanticAttentionFusion
from tda.models.gated_rgcn import GatedRGCN
from tda.topology.cache import compute_channel_topology_cached
from tda.train import (_acc_fn, _build_rgcn_edges, _f1_fn, _loss_fn,
                       _shuffle_topology_within_class_and_split)
from tda.utils import Timer, get_device, load_json, save_json, set_seed


def _manual_topology(bundle, config, device, verbose):
    """전 base relation 채널의 node PI (topo_seed 고정 + 캐시 → 데이터셋당 1회)."""
    gt = config["gated"]
    topo_seed = int(gt.get("topo_seed", 777))
    cache_dir = gt.get("topo_cache", "cache/topo_manual")
    feats = []
    for rname, adj in bundle.base_relations.items():
        arr = compute_channel_topology_cached(
            adj, config, topo_seed, device=device, verbose=verbose,
            cache_dir=cache_dir, tag=f"gated__{config['dataset']}__{rname}")
        feats.append(torch.tensor(arr, dtype=torch.float32, device=device))
    return feats


def run_gated(config: dict, dataset: str, data_root: str, device=None,
              output_dir=None, verbose: bool = True) -> Dict:
    gt = config["gated"]
    injection = gt.get("injection", "none")      # none | concat | gate
    content = gt.get("content", "real")          # real | noise | mix (injection=none 이면 무시)
    assert injection in ("none", "concat", "gate") and content in ("real", "noise", "mix")
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = device or get_device()
    bundle = get_dataset(dataset, config, data_root)
    x = bundle.x.to(device)
    y = bundle.y.to(device)
    masks = {k: v.to(device) for k, v in bundle.masks.items()}
    ei, et, nrel = _build_rgcn_edges(bundle, device)
    n = bundle.num_nodes
    record = {"dataset": dataset, "task": "gated_nc", "seed": seed,
              "injection": injection, "content": content if injection != "none" else None,
              "num_nodes": n, "num_classes": bundle.num_classes}

    # ---- 위상 특징 준비 (조건별) ----
    fusion, fused_builder, topo_dim = None, None, 0
    if injection != "none":
        pc = config["pdgnn"]
        topo_dim = pc["pi_resolution"] ** 2 * pc["hks_K"]
        if content == "noise":
            channels = [torch.randn(n, topo_dim, device=device)]
        else:
            with Timer("gated_topology"):
                channels = _manual_topology(bundle, config, device, verbose)
            if content == "mix":  # NC class-mix 와 동일 셔플 (class·split 내, 채널 공통 순열)
                channels, mix_stats = _shuffle_topology_within_class_and_split(
                    channels, y, masks, bundle.multilabel, seed, config)
                record["class_wise_mixing"] = mix_stats
        fusion = SemanticAttentionFusion(topo_dim).to(device)

        def fused_builder():
            return fusion(channels)

    # ---- 모델 (모든 조건 동일 구현; gate 조건만 게이트 경로 사용) ----
    gc_ = gt
    in_dim = x.size(1) + (topo_dim if injection == "concat" else 0)
    model = GatedRGCN(in_dim=in_dim, hidden_dim=gc_.get("hidden", 64),
                      num_classes=bundle.num_classes, num_relations=nrel,
                      num_layers=gc_.get("num_layers", 2), dropout=gc_.get("dropout", 0.5),
                      gate_dim=(topo_dim if injection == "gate" else None),
                      gate_hidden=gc_.get("gate_hidden", 64)).to(device)
    params = list(model.parameters())
    if fusion is not None:
        params += list(fusion.parameters())
    opt = torch.optim.Adam(params, lr=gc_.get("lr", 0.01),
                           weight_decay=gc_.get("weight_decay", 5e-4))

    def forward():
        if injection == "concat":
            return model(torch.cat([x, fused_builder()], dim=1), ei, et)
        if injection == "gate":
            return model(x, ei, et, gate_feat=fused_builder())
        return model(x, ei, et)

    best_val, best_state = -1.0, None
    for ep in range(1, gc_.get("epochs", 100) + 1):
        model.train()
        opt.zero_grad()
        loss = _loss_fn(forward(), y, masks["train"], bundle.multilabel)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vf1 = _f1_fn(forward(), y, masks["val"], bundle.multilabel)
        if vf1 > best_val:
            best_val = vf1
            best_state = (copy.deepcopy(model.state_dict()),
                          copy.deepcopy(fusion.state_dict()) if fusion else None)
        if verbose and (ep % 20 == 0 or ep == 1):
            print(f"  [gated] ep {ep:3d} loss={loss.item():.4f} val_f1={vf1:.4f}", flush=True)
    model.load_state_dict(best_state[0])
    if fusion is not None and best_state[1] is not None:
        fusion.load_state_dict(best_state[1])
    model.eval()
    with torch.no_grad():
        logits = forward()
        record.update({
            "val_macro_f1": float(best_val),
            "test_macro_f1": float(_f1_fn(logits, y, masks["test"], bundle.multilabel)),
            "test_accuracy": float(_acc_fn(logits, y, masks["test"], bundle.multilabel))})
    print(f"[result] gated {dataset} inj={injection} cont={record['content']} "
          f"test_f1={record['test_macro_f1']:.4f}", flush=True)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_json(config, os.path.join(output_dir, "config.json"))
        save_json(record, os.path.join(output_dir, "metrics.json"))
        print(f"[saved] {output_dir}/metrics.json", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    config = load_json(args.config)
    config["dataset"] = args.dataset or config["dataset"]
    if args.seed is not None:
        config["seed"] = args.seed
    device = torch.device("cpu") if args.cpu else get_device()
    run_gated(config, config["dataset"], args.data_root, device=device,
              output_dir=args.output_dir)


if __name__ == "__main__":
    main()
