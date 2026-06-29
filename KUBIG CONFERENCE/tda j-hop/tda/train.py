"""Staged 학습 드라이버 (플랜 Option A).

  Stage 1: GTN 으로 메타패스 채널 발견 + 노드 분류 학습 -> 채널 인접행렬 H (C,N,N)
  Stage 2: 채널마다 PDGNN 위상 특징 (N, res^2 * K) 추출 (PDGNN frozen)
  Stage 3: semantic attention 으로 채널 위상 특징 융합 -> 타겟 특징 강화 -> HAN 분류

use_topology=False 면 Stage 1/2 를 건너뛰고 HAN 단독(baseline)으로 학습한다.
실험은 데이터셋(--dataset)과 config 플래그만 바꿔 동일 코드로 실행한다.

사용:
  python -m tda.train --config configs/acm.json --dataset acm \
      --data-root <hgb_root> --output-dir runs/acm_full
"""
from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from tda.data import get_dataset
from tda.models.fusion import SemanticAttentionFusion
from tda.models.gtn import GTN, GTConv
from tda.models.han import HAN
from tda.topology.epd import compute_channel_topology
from tda.utils import (Timer, accuracy, get_device, load_json, macro_f1,
                       multilabel_accuracy, multilabel_macro_f1, save_json, set_seed)


def _loss_fn(logits, y, mask, multilabel):
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits[mask], y[mask].float())
    return F.cross_entropy(logits[mask], y[mask])


def _f1_fn(logits, y, mask, multilabel):
    return (multilabel_macro_f1(logits[mask], y[mask]) if multilabel
            else macro_f1(logits[mask], y[mask]))


def _acc_fn(logits, y, mask, multilabel):
    return (multilabel_accuracy(logits[mask], y[mask]) if multilabel
            else accuracy(logits[mask], y[mask]))


def _build_A_stack(bundle, device) -> torch.Tensor:
    """기저 관계 이진 인접 + 항등행렬 -> (R+1, N, N) GTN 입력 스택."""
    rels = [bundle.base_relations[k] for k in bundle.base_relations]
    n = bundle.num_nodes
    rels.append(torch.eye(n))
    return torch.stack(rels, dim=0).to(device)


def _build_edge_index_dict(bundle, device) -> Dict:
    return {(bundle.target, mp, bundle.target): ei.to(device)
            for mp, ei in bundle.han_metapaths.items()}


def train_gtn(bundle, A_stack, x, y, masks, config, device, verbose=False) -> torch.Tensor:
    """Stage 1: GTN 학습. 최적 val 시점의 채널 인접행렬 H (C,N,N, detached) 반환."""
    gc = config["gtn"]
    model = GTN(num_relations=A_stack.size(0), num_channels=gc["num_channels"],
                in_dim=x.size(1), hidden_dim=gc["hidden_dim"], num_classes=bundle.num_classes,
                num_layers=gc["num_layers"], dropout=gc["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    best_val, best_state = -1.0, None
    for ep in range(1, gc["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(A_stack, x)
        loss = _loss_fn(logits, y, masks["train"], bundle.multilabel)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(A_stack, x)
            vf1 = _f1_fn(logits, y, masks["val"], bundle.multilabel)
        if vf1 > best_val:
            best_val, best_state = vf1, copy.deepcopy(model.state_dict())
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  [gtn] ep {ep:3d} loss={loss.item():.4f} val_f1={vf1:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(A_stack, x)
        H = model.discover(A_stack).detach()
    # GTN 단독 분류 성능(= 실험 A3, EPD 없이 GTN 만)도 기록
    train_gtn.last_val_f1 = float(best_val)
    train_gtn.last_test_f1 = _f1_fn(logits, y, masks["test"], bundle.multilabel)
    train_gtn.last_test_acc = _acc_fn(logits, y, masks["test"], bundle.multilabel)
    print(f"  [gtn] best val_f1={best_val:.4f} test_f1={train_gtn.last_test_f1:.4f} (A3=GTN-only); "
          f"discovered H {tuple(H.shape)}", flush=True)
    # 발견된 메타패스 해석용 어텐션 가중치 저장
    train_gtn.last_attentions = [[w.cpu().tolist() for w in layer]
                                 for layer in model.channel_attentions()]
    return H


def train_han(bundle, x_in_builder, in_dim, edge_index_dict, y, masks, config,
              device, extra_params=None, extra_eval=None, verbose=False) -> Dict:
    """Stage 3: HAN(+fusion) 학습. test macro-F1/accuracy 반환.

    x_in_builder(): 매 forward 마다 입력 특징을 만드는 콜러블 (fusion 학습 반영).
    extra_params: fusion 등 함께 최적화할 추가 파라미터 리스트.
    extra_eval: {name: builder} — 최적 모델로 추가 test 평가(예: permutation 진단).
    """
    hc = config["han"]
    model = HAN(in_dim=in_dim, hidden_dim=hc["hidden_dim"], num_classes=bundle.num_classes,
                target=bundle.target, metapath_edge_types=list(edge_index_dict.keys()),
                heads=hc["heads"], dropout=hc["dropout"]).to(device)
    params = list(model.parameters())
    for m in (extra_params or []):
        params += list(m.parameters())
    opt = torch.optim.Adam(params, lr=hc["lr"], weight_decay=hc["weight_decay"])
    best_val, best_state = -1.0, None
    for ep in range(1, hc["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_in_builder(), edge_index_dict)
        loss = _loss_fn(logits, y, masks["train"], bundle.multilabel)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_in_builder(), edge_index_dict)
            vf1 = _f1_fn(logits, y, masks["val"], bundle.multilabel)
        if vf1 > best_val:
            best_val = vf1
            best_state = (copy.deepcopy(model.state_dict()),
                          [copy.deepcopy(m.state_dict()) for m in extra_params] if extra_params else None)
        if verbose and (ep % 20 == 0 or ep == 1):
            print(f"  [han] ep {ep:3d} loss={loss.item():.4f} val_f1={vf1:.4f}", flush=True)
    model.load_state_dict(best_state[0])
    if extra_params and best_state[1] is not None:
        for m, s in zip(extra_params, best_state[1]):
            m.load_state_dict(s)
    model.eval()
    with torch.no_grad():
        logits = model(x_in_builder(), edge_index_dict)
        ml = bundle.multilabel
        result = {"val_macro_f1": float(best_val),
                  "test_macro_f1": float(_f1_fn(logits, y, masks["test"], ml)),
                  "test_accuracy": float(_acc_fn(logits, y, masks["test"], ml))}
        for name, builder in (extra_eval or {}).items():
            lg = model(builder(), edge_index_dict)
            result[f"test_macro_f1_{name}"] = float(_f1_fn(lg, y, masks["test"], ml))
    return result


def run(config: dict, dataset: str, data_root: str, device=None,
        output_dir: Optional[str] = None, verbose: bool = True) -> Dict:
    seed = config.get("seed", 0)
    set_seed(seed)
    device = device or get_device()
    use_topo = config.get("use_topology", True)
    print(f"[run] dataset={dataset} use_topology={use_topo} device={device} seed={seed}", flush=True)

    bundle = get_dataset(dataset, config, data_root)
    x = bundle.x.to(device)
    y = bundle.y.to(device)
    masks = {k: v.to(device) for k, v in bundle.masks.items()}
    edge_index_dict = _build_edge_index_dict(bundle, device)
    print(f"[data] N={bundle.num_nodes} feat={x.size(1)} classes={bundle.num_classes} "
          f"base_rel={list(bundle.base_relations)} han_mp={list(bundle.han_metapaths)}", flush=True)

    topo_source = config.get("topology_source", "gtn")  # 'gtn' (C) | 'manual' (B)
    record = {"dataset": dataset, "use_topology": use_topo, "seed": seed,
              "topology_source": topo_source if use_topo else None,
              "num_nodes": bundle.num_nodes, "num_classes": bundle.num_classes}

    if use_topo:
        # Stage 1: 채널(=메타패스) 인접행렬 확보 — GTN 자동발견(C) 또는 고정 수동메타패스(B).
        if topo_source == "gtn":
            with Timer("stage1_gtn"):
                A_stack = _build_A_stack(bundle, device)
                H = train_gtn(bundle, A_stack, x, y, masks, config, device, verbose)
            channel_adjs = [H[c] for c in range(H.size(0))]
            record["gtn_attentions"] = getattr(train_gtn, "last_attentions", None)
            # 실험 A3: EPD 없이 GTN 단독 분류 성능
            record["gtn_only_test_macro_f1"] = getattr(train_gtn, "last_test_f1", None)
            record["gtn_only_test_accuracy"] = getattr(train_gtn, "last_test_acc", None)
            record["gtn_only_val_macro_f1"] = getattr(train_gtn, "last_val_f1", None)
        elif topo_source == "manual":
            manual = config.get("manual_metapaths", config["han_metapaths"])
            channel_adjs = [bundle.base_relations[m].to(device) for m in manual]
            record["manual_metapaths"] = list(manual)
            print(f"  [manual] 고정 메타패스 채널: {list(manual)}", flush=True)
        elif topo_source == "random":
            # D5: 학습 없이 무작위 초기화 GTN 으로 채널 발견 (random meta-path 대조군)
            gc = config["gtn"]
            A_stack = _build_A_stack(bundle, device)
            gtn = GTN(num_relations=A_stack.size(0), num_channels=gc["num_channels"],
                      in_dim=x.size(1), hidden_dim=gc["hidden_dim"], num_classes=bundle.num_classes,
                      num_layers=gc["num_layers"], dropout=gc["dropout"]).to(device)
            for m in gtn.modules():
                if isinstance(m, GTConv):
                    torch.nn.init.normal_(m.weight)
            with torch.no_grad():
                H = gtn.discover(A_stack).detach()
            channel_adjs = [H[c] for c in range(H.size(0))]
            record["random_metapath"] = True
            print(f"  [random] 무작위 메타패스 채널 {H.size(0)}개 (학습 없음)", flush=True)
        else:
            raise ValueError(f"unknown topology_source '{topo_source}' (gtn|manual|random)")

        with Timer("stage2_pdgnn_topology"):
            topo_channels: List[torch.Tensor] = []
            for c, adj in enumerate(channel_adjs):
                feat = compute_channel_topology(adj, config, seed, device=device, verbose=verbose)
                topo_channels.append(torch.tensor(feat, dtype=torch.float32, device=device))
                print(f"  [stage2] channel {c} topo {tuple(topo_channels[-1].shape)}", flush=True)
            topo_dim = topo_channels[0].size(1)

        fusion = SemanticAttentionFusion(topo_dim).to(device)

        # 진단: node_features=off → 위상만으로 분류(topology-only). 기본은 'on'.
        nf = config.get("node_features", "on")
        x_base = x if nf == "on" else torch.zeros_like(x)
        record["node_features"] = nf

        def x_in_builder():
            return torch.cat([x_base, fusion(topo_channels)], dim=1)  # (원본 또는 0) ⊕ 융합 위상

        # 진단: permutation test — test 시 위상 특징 행을 섞어 모델이 위상을 실제 쓰는지 확인.
        extra_eval = None
        if config.get("permute_topology", False):
            perm = torch.randperm(bundle.num_nodes, device=device)
            extra_eval = {"permuted": lambda: torch.cat([x_base, fusion(topo_channels)[perm]], dim=1)}

        with Timer("stage3_han"):
            metrics = train_han(bundle, x_in_builder, x.size(1) + topo_dim, edge_index_dict,
                                y, masks, config, device, extra_params=[fusion],
                                extra_eval=extra_eval, verbose=verbose)
        record["fusion_beta"] = getattr(fusion, "last_beta", torch.zeros(0)).cpu().tolist()
        record["topo_dim"] = topo_dim
    else:
        nf = config.get("node_features", "on")
        x_base = x if nf == "on" else torch.zeros_like(x)
        record["node_features"] = nf
        with Timer("stage3_han_baseline"):
            metrics = train_han(bundle, lambda: x_base, x.size(1), edge_index_dict,
                                y, masks, config, device, extra_params=None, verbose=verbose)

    record.update(metrics)
    print(f"[result] test_macro_f1={metrics['test_macro_f1']:.4f} "
          f"test_acc={metrics['test_accuracy']:.4f} val_f1={metrics['val_macro_f1']:.4f}", flush=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_json(config, os.path.join(output_dir, "config.json"))
        save_json(record, os.path.join(output_dir, "metrics.json"))
        print(f"[saved] {output_dir}/metrics.json", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None, help="config 의 dataset 을 덮어씀")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-topology", action="store_true", help="baseline A1 (HAN 단독)")
    ap.add_argument("--topology-source", choices=["gtn", "manual", "random"], default=None,
                    help="gtn=C(GTN 발견) | manual=B(고정 수동) | random=D5(무작위) 메타패스")
    ap.add_argument("--node-features", choices=["on", "off"], default=None,
                    help="off=topology-only 진단")
    ap.add_argument("--permute-topology", action="store_true", help="permutation 진단 평가 추가")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    config = load_json(args.config)
    dataset = args.dataset or config.get("dataset", "acm")
    if args.no_topology:
        config["use_topology"] = False
    if args.topology_source is not None:
        config["topology_source"] = args.topology_source
    if args.node_features is not None:
        config["node_features"] = args.node_features
    if args.permute_topology:
        config["permute_topology"] = True
    if args.seed is not None:
        config["seed"] = args.seed
    device = torch.device("cpu") if args.cpu else get_device()
    run(config, dataset, os.path.abspath(args.data_root), device, args.output_dir)


if __name__ == "__main__":
    main()
