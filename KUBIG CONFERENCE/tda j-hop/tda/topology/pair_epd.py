"""Pair-vicinity EPD (LP Level 2) — TLC-GNN 의 pair 위상을 PDGNN 근사로.

노드쌍 (u,v)의 vicinity = ego_hop(u) ∪ ego_hop(v) (train 엣지 그래프) 위에서 HKS 필터
EPD 를 계산한다. 정확 EPD(gudhi)는 PDGNN 학습 라벨 샘플에만, 전체 후보쌍은 PDGNN 추론.
split_seed·topo_seed 고정 전제 하에 결과를 디스크 캐시(데이터셋당 1회 계산).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch

from tda.topology.epd import _exact_epd, train_pdgnn
from tda.topology.hks import compute_hks
from tda.topology.persistence_image import PersistenceImage


def nx_from_dense(adj: torch.Tensor) -> nx.Graph:
    a = adj.detach().cpu().numpy()
    g = nx.Graph()
    g.add_nodes_from(range(a.shape[0]))
    src, dst = np.nonzero(np.triu(a, k=1))
    g.add_edges_from(zip(src.tolist(), dst.tolist()))
    return g


def _pair_vicinity(g: nx.Graph, u: int, v: int, hop: int, node_filt: dict, max_nodes: int):
    """(filt (m,), edge_index (2,E)) — ego(u)∪ego(v) 지역 재인덱싱. u,v 는 cap 시에도 보존."""
    nodes = set(nx.ego_graph(g, u, radius=hop).nodes()) | set(nx.ego_graph(g, v, radius=hop).nodes())
    if len(nodes) > max_nodes:
        keep = sorted(nodes, key=lambda nd: node_filt.get(nd, 0.0))[:max_nodes]
        nodes = set(keep) | {u, v}
    sub = g.subgraph(nodes)
    order = list(sub.nodes())
    remap = {nd: i for i, nd in enumerate(order)}
    filt = np.array([node_filt.get(nd, 0.0) for nd in order], dtype=np.float64)
    if sub.number_of_edges() == 0:
        return filt, np.zeros((2, 0), dtype=np.int64)
    e = np.array([(remap[a], remap[b]) for a, b in sub.edges()], dtype=np.int64).T
    return filt, np.concatenate([e, e[[1, 0]]], axis=1)


def _pair_samples(g, hks, pairs, hop, max_nodes, n_samples, seed):
    """PDGNN 학습 샘플: 샘플 쌍의 vicinity + 정확 EPD 라벨 (스케일별)."""
    rng = np.random.RandomState(seed)
    K = hks.shape[1]
    idx = rng.choice(len(pairs), size=min(n_samples, len(pairs)), replace=False)
    filts_by_k = [{nd: float(hks[nd, k]) for nd in g.nodes()} for k in range(K)]
    samples = []
    for i in idx:
        u, v = int(pairs[i][0]), int(pairs[i][1])
        for k in range(K):
            filt, ei = _pair_vicinity(g, u, v, hop, filts_by_k[k], max_nodes)
            if ei.shape[1] == 0:
                continue
            gt = _exact_epd(filt, ei)
            if gt.size == 0:
                continue
            samples.append((filt.reshape(-1, 1).astype(np.float32),
                            ei.astype(np.int64), gt.astype(np.float32)))
    return samples


@torch.no_grad()
def _predict_pairs(model, g, hks, pairs, hop, max_nodes, imager, device) -> np.ndarray:
    K = hks.shape[1]
    R2 = imager.dim
    out = np.zeros((len(pairs), R2 * K), dtype=np.float64)
    filts_by_k = [{nd: float(hks[nd, k]) for nd in g.nodes()} for k in range(K)]
    model.eval()
    for i, (u, v) in enumerate(pairs):
        for k in range(K):
            filt, ei = _pair_vicinity(g, int(u), int(v), hop, filts_by_k[k], max_nodes)
            if ei.shape[1] == 0:
                continue
            ft = torch.tensor(filt.reshape(-1, 1).astype(np.float32), device=device)
            et = torch.tensor(ei, device=device)
            pred = model(ft, et).cpu().numpy()
            pred = pred[pred[:, 1] > pred[:, 0]]
            if pred.size:
                out[i, k * R2:(k + 1) * R2] = imager.transform(pred)
    return out


def compute_pair_topology_cached(adj: torch.Tensor, pairs: np.ndarray, config: dict,
                                 topo_seed: int, cache_dir: str, tag: str,
                                 device=None, verbose: bool = False) -> np.ndarray:
    """(E, res²·K) pair PI. (train 그래프 + pdgnn 설정 + 쌍 목록 + seed) 해시로 디스크 캐시."""
    pc = config["pdgnn"]
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    key_src = hashlib.sha1()
    key_src.update(np.ascontiguousarray(adj.detach().cpu().numpy()).tobytes())
    key_src.update(np.ascontiguousarray(pairs.astype(np.int64)).tobytes())
    key_src.update(json.dumps(pc, sort_keys=True, default=str).encode())
    key_src.update(f"{tag}|{int(topo_seed)}".encode())
    key = key_src.hexdigest()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(path):
            z = np.load(path)
            if np.array_equal(z["pairs"], pairs):
                if verbose:
                    print(f"[pair_epd] cache hit {key[:12]}", flush=True)
                return z["feats"]
    g = nx_from_dense(adj)
    from tda.topology.epd import _edge_index_of
    hks = compute_hks(_edge_index_of(g), num_nodes=adj.size(0), K=pc["hks_K"],
                      device=dev, verbose=verbose)
    samples = _pair_samples(g, hks, pairs, pc["hop"], pc["max_nodes"],
                            pc["n_train_samples"], topo_seed)
    model = train_pdgnn(samples, hidden=pc["hidden_dim"], layers=pc["layers"],
                        epochs=pc["epochs"], lr=pc["lr"], seed=topo_seed + 1234,
                        device=dev, verbose=verbose, agg=pc.get("agg", "sum_min"))
    imager = PersistenceImage(resolution=pc["pi_resolution"], sigma=pc.get("pi_sigma", 0.5),
                              birth_range=tuple(pc.get("pi_birth_range", (-3.0, 3.0))),
                              pers_range=tuple(pc.get("pi_pers_range", (0.0, 6.0))))
    feats = _predict_pairs(model, g, hks, pairs, pc["hop"], pc["max_nodes"], imager, dev)
    if cache_dir:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, pairs=pairs, feats=feats)
        os.replace(tmp, path)
        if verbose:
            print(f"[pair_epd] computed+cached {key[:12]} feats={feats.shape}", flush=True)
    return feats
