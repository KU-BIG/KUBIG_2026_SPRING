"""Link Prediction (Level 1) — node-level 위상 특징의 LP 기여 검증. docs/lp_design.md 참조.

과제: 주 타겟-타겟 base relation(acm=PAP 등)의 엣지 예측.
  - split_seed 고정 → 모든 run 이 동일 분할 (seed 는 모델 init/negative sampling 만 변화)
  - 누출 방지: val/test positive 엣지는 메시지패싱 인접과 위상 채널 **모두에서 제거**
  - 위상: manual 채널(GTN 생략) + topo_seed 고정 → 데이터셋당 1회 계산(캐시 공유)
  - 조건: use_topology=false(a) | topology_noise=random(b1) | original(c) | degree_mix(m)
    (+ lp.topo_exclude_target: 위상 채널에서 예측 대상 관계 제외 — 신호 출처 분해용 c')

사용:
  python -m tda.lp --config configs/campaign/acm__lp_c.json --dataset acm \
      --data-root <root> --output-dir runs/lp/acm_c_s0 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tda.data import get_dataset
from tda.models.fusion import SemanticAttentionFusion
from tda.models.rgcn import RGCN
from tda.topology.cache import compute_channel_topology_cached
from tda.utils import Timer, get_device, load_json, save_json, set_seed


# ---------- 평가 지표 (의존성 없는 구현) ----------

def auc_score(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """ROC-AUC (Mann-Whitney U), tie 는 평균 랭크 처리."""
    from scipy.stats import rankdata

    s = torch.cat([pos, neg]).detach().cpu().numpy()
    n_pos, n_neg = len(pos), len(neg)
    rank = rankdata(s)  # average ranks
    return float((rank[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ap_score(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Average Precision."""
    s = torch.cat([pos, neg]).detach().cpu()
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    y = y[s.argsort(descending=True)]
    cum_pos = torch.cumsum(y, 0)
    prec = cum_pos / torch.arange(1, len(y) + 1, dtype=torch.float)
    return float((prec * y).sum() / max(1, int(y.sum())))


# ---------- 엣지 분할 / negative sampling ----------

def split_edges(A: torch.Tensor, cap: int, split_seed: int):
    """대칭 이진 인접 A 의 (u<v) 엣지를 [cap 이하 샘플 후] 80/10/10 분할.
    반환: train/val/test (E,2) long, 그리고 전체 positive set(음성 거부용)."""
    ut = torch.triu(A, diagonal=1).nonzero(as_tuple=False)  # (E,2), u<v
    rng = np.random.RandomState(split_seed)
    idx = rng.permutation(ut.size(0))
    if cap and ut.size(0) > cap:
        idx = idx[:cap]
    e = ut[torch.from_numpy(np.ascontiguousarray(idx)).long()]
    n = e.size(0)
    n_val, n_test = int(n * 0.1), int(n * 0.1)
    val, test, train = e[:n_val], e[n_val:n_val + n_test], e[n_val + n_test:]
    pos_set = {(int(u), int(v)) for u, v in ut.tolist()}  # cap 밖 엣지도 negative 금지
    return train, val, test, pos_set


def sample_negatives(n: int, num_nodes: int, forbid: set, rng: np.random.RandomState) -> torch.Tensor:
    out = []
    seen = set(forbid)
    while len(out) < n:
        u = rng.randint(0, num_nodes, size=n * 2)
        v = rng.randint(0, num_nodes, size=n * 2)
        for a, b in zip(u, v):
            if a == b:
                continue
            key = (int(min(a, b)), int(max(a, b)))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= n:
                break
    return torch.tensor(out, dtype=torch.long)


# ---------- 위상 특징 (manual 채널, 캐시) ----------

def degree_mix(topo: List[torch.Tensor], deg: torch.Tensor, seed: int, n_bucket: int = 10):
    """degree 분위 bucket 내에서 노드 PI 행 셔플(derangement 지향, 모든 채널 동일 순열).
    class-mix(NC)의 LP 판 — '차수로 설명되는 것 이상'의 per-node 위상 정보를 격리."""
    n = deg.size(0)
    order = deg.argsort()
    perm = torch.arange(n)
    gen = torch.Generator().manual_seed(int(seed) + 4241)
    bounds = [int(round(i * n / n_bucket)) for i in range(n_bucket + 1)]
    for i in range(n_bucket):
        idxs = order[bounds[i]:bounds[i + 1]]
        if idxs.numel() < 2:
            continue
        shuffled = idxs[torch.randperm(idxs.numel(), generator=gen)]
        perm[shuffled] = shuffled.roll(1)
    return [t[perm] for t in topo], perm


def build_topology(bundle, mp_rels: Dict[str, torch.Tensor], config: dict, device, verbose):
    """manual 채널(= mp_rels 의 각 관계)별 node PI. topo_seed 고정 + 캐시 → 데이터셋당 1회 계산."""
    lp = config["lp"]
    topo_seed = int(lp.get("topo_seed", 777))
    cache_dir = lp.get("topo_cache", "cache/topo_lp")
    names = [r for r in mp_rels
             if not (lp.get("topo_exclude_target", False) and r == lp["target_relation"])]
    feats = []
    for rname in names:
        arr = compute_channel_topology_cached(
            mp_rels[rname], config, topo_seed, device=device, verbose=verbose,
            cache_dir=cache_dir, tag=f"lp__{config['dataset']}__{rname}")
        feats.append(torch.tensor(arr, dtype=torch.float32, device=device))
    return feats, names


# ---------- LP 모델 ----------

class LPHead(nn.Module):
    """대칭 pair 스코어: MLP([z_u⊙z_v, |z_u−z_v| (⊕ pair 위상)]).
    pair_dim>0 이면 TLC-GNN 식으로 pair 특징을 decoder 에 concat (LP Level 2)."""

    def __init__(self, dim: int, hidden: int = 64, pair_dim: int = 0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim + pair_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor, pairs: torch.Tensor, pf: torch.Tensor = None) -> torch.Tensor:
        zu, zv = z[pairs[:, 0]], z[pairs[:, 1]]
        h = torch.cat([zu * zv, (zu - zv).abs()], dim=-1)
        if pf is not None:
            h = torch.cat([h, pf], dim=-1)
        return self.mlp(h).squeeze(-1)


def cn_counts(A: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """train 그래프 기준 공통이웃 수 (pair 별)."""
    return (A[pairs[:, 0]] * A[pairs[:, 1]]).sum(1)


def cn_bucket_mix(feats: torch.Tensor, cn: torch.Tensor, seed: int) -> torch.Tensor:
    """CN bucket(0/1/2/3+) 내에서 pair 특징 행 셔플 — 'CN 으로 설명되는 것 이상'을 격리."""
    bucket = cn.clamp(max=3).long()
    perm = torch.arange(len(feats))
    gen = torch.Generator().manual_seed(int(seed) + 9173)
    for b in range(4):
        idx = (bucket == b).nonzero(as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue
        shuffled = idx[torch.randperm(idx.numel(), generator=gen)]
        perm[shuffled] = shuffled.roll(1)
    return feats[perm]


def _build_edges(rels: Dict[str, torch.Tensor], device):
    eis, ets = [], []
    for ridx, name in enumerate(rels):
        ei = rels[name].nonzero(as_tuple=False).t().contiguous()
        eis.append(ei)
        ets.append(torch.full((ei.size(1),), ridx, dtype=torch.long))
    return torch.cat(eis, 1).to(device), torch.cat(ets, 0).to(device), len(rels)


# ---------- 메인 ----------

def run_lp(config: dict, dataset: str, data_root: str, device=None,
           output_dir=None, verbose: bool = True) -> Dict:
    lp = config["lp"]
    seed = int(config.get("seed", 0))          # 모델 init / train negative 만 좌우
    set_seed(seed)
    device = device or get_device()
    bundle = get_dataset(dataset, config, data_root)
    x = bundle.x.to(device)
    n = bundle.num_nodes
    tgt_rel = lp["target_relation"]
    assert tgt_rel in bundle.base_relations, f"{tgt_rel} not in base_relations"

    # 1) 고정 분할 (+음성 val/test 도 split_seed 로 고정 → run 간 동일 평가셋)
    A_tgt = bundle.base_relations[tgt_rel]
    train_pos, val_pos, test_pos, pos_set = split_edges(
        A_tgt, int(lp.get("edge_cap", 5000)), int(lp.get("split_seed", 20260705)))
    srng = np.random.RandomState(int(lp.get("split_seed", 20260705)) + 1)
    val_neg = sample_negatives(len(val_pos), n, pos_set, srng)
    test_neg = sample_negatives(len(test_pos), n, pos_set | {tuple(e) for e in val_neg.tolist()}, srng)

    # 2) 누출 방지: 타겟 관계 인접을 train 엣지로만 재구성 (메시지패싱·위상 공통 사용)
    A_train = torch.zeros_like(A_tgt)
    A_train[train_pos[:, 0], train_pos[:, 1]] = 1.0
    A_train[train_pos[:, 1], train_pos[:, 0]] = 1.0
    mp_rels = {k: (A_train if k == tgt_rel else v) for k, v in bundle.base_relations.items()}

    record = {"dataset": dataset, "task": "lp", "seed": seed, "target_relation": tgt_rel,
              "n_train_pos": len(train_pos), "n_val_pos": len(val_pos), "n_test_pos": len(test_pos),
              "use_topology": bool(config.get("use_topology", True)),
              "topology_noise": config.get("topology_noise"),
              "topology_mode": config.get("topology_mode", "original"),
              "pair_feature": lp.get("pair_feature", "none")}

    # ---- LP Level 2: pair-vicinity 위상 (TLC-GNN 식 decoder 주입) ----
    pair_kind = lp.get("pair_feature", "none")          # none | real | noise | mix
    pair_feats, pair_dim, tn_fixed = None, 0, None
    if pair_kind != "none" or lp.get("fixed_train_neg", False):
        # 고정 train negative pool (특징 사전계산 위해; split_seed 로 고정 → run 간 동일.
        # L2 base 조건도 fixed_train_neg=true 로 같은 프로토콜을 씀)
        prng = np.random.RandomState(int(lp.get("split_seed", 20260705)) + 2)
        forbid = pos_set | {tuple(e) for e in val_neg.tolist()} | {tuple(e) for e in test_neg.tolist()}
        tn_fixed = sample_negatives(len(train_pos), n, forbid, prng)
    if pair_kind != "none":
        all_pairs = torch.cat([train_pos, tn_fixed, val_pos, val_neg, test_pos, test_neg])
        pc = config["pdgnn"]
        pair_dim = pc["pi_resolution"] ** 2 * pc["hks_K"]
        cn = cn_counts(A_train, all_pairs)
        record["test_auc_cn"] = auc_score(          # CN 휴리스틱 단독 baseline (진단)
            cn[-2 * len(test_pos):-len(test_pos)].float(), cn[-len(test_pos):].float())
        if pair_kind == "noise":
            pair_feats = torch.randn(len(all_pairs), pair_dim, device=device)
        else:
            from tda.topology.pair_epd import compute_pair_topology_cached
            with Timer("lp_pair_topology"):
                arr = compute_pair_topology_cached(
                    A_train, all_pairs.numpy(), config, int(lp.get("topo_seed", 777)),
                    lp.get("pair_cache", "cache/pair_epd"),
                    tag=f"lp2__{dataset}__{tgt_rel}", device=device, verbose=verbose)
            pair_feats = torch.tensor(arr, dtype=torch.float32, device=device)
            if pair_kind == "mix":
                pair_feats = cn_bucket_mix(pair_feats.cpu(), cn.cpu(), seed).to(device)
        # 구간 인덱스 (all_pairs 순서: tp | tn | vp | vn | sp | sn)
        ofs = np.cumsum([0, len(train_pos), len(tn_fixed), len(val_pos),
                         len(val_neg), len(test_pos), len(test_neg)])
        pf_slice = {k: pair_feats[ofs[i]:ofs[i + 1]]
                    for i, k in enumerate(["tp", "tn", "vp", "vn", "sp", "sn"])}

    # 3) 입력 특징 (조건 분기)
    fusion, topo_channels = None, []
    topo_dim = 0
    if config.get("use_topology", True):
        pc = config["pdgnn"]
        topo_dim = pc["pi_resolution"] ** 2 * pc["hks_K"]
        if config.get("topology_noise") == "random":                      # (b1)
            topo_channels = [torch.randn(n, topo_dim, device=device)]
            record["topo_kind"] = "noise"
        else:
            with Timer("lp_topology"):
                topo_channels, names = build_topology(bundle, mp_rels, config, device, verbose)
            record["topo_channels"] = names
            record["topo_kind"] = "real"
            if config.get("topology_mode", "original") == "degree_mix":   # (m)
                deg = A_train.sum(1)
                topo_channels, _ = degree_mix(topo_channels, deg, seed)
                record["topo_kind"] = "degree_mix"
        fusion = SemanticAttentionFusion(topo_dim).to(device)

    # 중요: topology 캐시(compute_channel_topology_cached)가 적중 시 전역 RNG 를 복원함 →
    # 모델 초기화가 seed 와 무관해지는 사고 방지 위해 run seed 로 재시드.
    set_seed(seed)

    def x_in():
        return torch.cat([x, fusion(topo_channels)], dim=1) if topo_channels else x

    # 4) encoder(RGCN, 임베딩 출력) + LP head 학습
    ei, et, nrel = _build_edges(mp_rels, device)
    hc = lp.get("hidden", 64)
    enc = RGCN(in_dim=x.size(1) + topo_dim, hidden_dim=hc, num_classes=hc,  # num_classes=출력 임베딩 차원
               num_relations=nrel, num_layers=int(lp.get("num_layers", 2)),
               dropout=float(lp.get("dropout", 0.5))).to(device)
    head = LPHead(hc, hidden=hc, pair_dim=pair_dim).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    if fusion is not None:
        params += list(fusion.parameters())
    opt = torch.optim.Adam(params, lr=float(lp.get("lr", 0.01)),
                           weight_decay=float(lp.get("weight_decay", 5e-4)))
    tp, vp, sp = train_pos.to(device), val_pos.to(device), test_pos.to(device)
    vn, sn = val_neg.to(device), test_neg.to(device)
    if tn_fixed is not None:
        tn_fixed = tn_fixed.to(device)

    def score(z, pairs, key):
        pf = pf_slice[key] if pair_kind != "none" else None
        return head(z, pairs, pf)

    nrng = np.random.RandomState(seed + 7)
    best_val, best_state = -1.0, None
    for ep in range(1, int(lp.get("epochs", 200)) + 1):
        enc.train(); head.train()
        # L2(pair 특징)에선 특징이 사전계산된 고정 pool 사용, L1 은 매 epoch 재샘플
        tn = tn_fixed if tn_fixed is not None \
            else sample_negatives(len(train_pos), n, pos_set, nrng).to(device)
        opt.zero_grad()
        z = enc(x_in(), ei, et)
        logits = torch.cat([score(z, tp, "tp"), score(z, tn, "tn")])
        y = torch.cat([torch.ones(len(tp), device=device), torch.zeros(len(tn), device=device)])
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            z = enc(x_in(), ei, et)
            v_auc = auc_score(score(z, vp, "vp"), score(z, vn, "vn"))
        if v_auc > best_val:
            best_val = v_auc
            best_state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(head.state_dict()),
                          copy.deepcopy(fusion.state_dict()) if fusion else None)
        if verbose and (ep % 50 == 0 or ep == 1):
            print(f"  [lp] ep {ep:3d} loss={loss.item():.4f} val_auc={v_auc:.4f}", flush=True)
    enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    if fusion is not None and best_state[2] is not None:
        fusion.load_state_dict(best_state[2])
    enc.eval(); head.eval()
    with torch.no_grad():
        z = enc(x_in(), ei, et)
        ps, ns = score(z, sp, "sp"), score(z, sn, "sn")
    record.update({"val_auc": float(best_val), "test_auc": auc_score(ps, ns),
                   "test_ap": ap_score(ps, ns)})
    print(f"[result] lp {dataset} kind={record.get('topo_kind','none')} "
          f"test_auc={record['test_auc']:.4f} test_ap={record['test_ap']:.4f}", flush=True)
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
    run_lp(config, config["dataset"], args.data_root, device=device,
           output_dir=args.output_dir)


if __name__ == "__main__":
    main()
