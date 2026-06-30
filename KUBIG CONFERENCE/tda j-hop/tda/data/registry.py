"""데이터셋 레지스트리 + 기저 관계/메타패스 구성.

새 데이터셋 추가 = (1) 로더 함수 작성 후 `DATASETS` 에 등록, (2) `configs/<name>.json`
에 base_relations / han_metapaths 정의. 그러면 `--dataset <name>` 으로 동일 실험 실행.

`HeteroBundle` 은 파이프라인이 쓰는 모든 텐서를 담는다:
  - x, y, masks, num_classes : 타겟 노드 분류용
  - base_relations : 타겟->타겟 (N,N) 이진 인접행렬들 (GTN 의 기저 관계 스택 A)
  - han_metapaths  : 타겟->타겟 edge_index (HAN 의 메타패스 그래프, kNN 희소화)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import scipy.sparse as sp
import torch

from tda.data.acm import load_acm
from tda.data.big import (load_aifb, load_am, load_aminer, load_bgs, load_mag, load_mutag,
                          load_pubmed_hne, load_yelp_hne)
from tda.data.hgb import load_dblp, load_dblp_pyg, load_freebase, load_imdb, load_imdb_pyg
from tda.data.toy import load_toy

# name -> loader() returning (PyG HeteroData, target_node_type)
DATASETS: Dict[str, Callable] = {
    "acm": load_acm,        # 학술/인용
    "dblp": load_dblp,      # 학술/인용
    "imdb": load_imdb,      # 영화 (멀티라벨)
    "freebase": load_freebase,  # 지식그래프
    "aminer": load_aminer,  # 학술 (대형, 이종 subsample)
    "mag": load_mag,        # 학술 (ogbn-mag, 대형, 이종 subsample)
    "dblp_pyg": load_dblp_pyg,  # 학술 (PyG/MAGNN 판)
    "imdb_pyg": load_imdb_pyg,  # 영화 (PyG/MAGNN 판, 단일라벨 3클래스)
    "aifb": load_aifb,      # RDF/연구기관 (다관계, featureless)
    "mutag": load_mutag,    # RDF/화학 (다관계, featureless)
    "bgs": load_bgs,        # RDF/지질 (대형, 다관계, featureless)
    "am": load_am,          # RDF/박물관 (초대형, 다관계, featureless)
    "pubmed": load_pubmed_hne,  # HNE/생의학 (GENE/DISEASE/CHEMICAL/SPECIES, DISEASE 8클래스)
    "yelp": load_yelp_hne,      # HNE/business (BUSINESS 16클래스 멀티라벨, featureless)
    "toy": load_toy,
}


def list_datasets() -> List[str]:
    return sorted(DATASETS.keys())


@dataclass
class HeteroBundle:
    name: str
    target: str
    x: torch.Tensor                       # (N, F) 타겟 노드 특징
    y: torch.Tensor                       # (N,)
    masks: Dict[str, torch.Tensor]        # train/val/test bool (N,)
    num_classes: int
    base_relations: Dict[str, torch.Tensor]   # name -> (N, N) dense 이진 인접
    han_metapaths: Dict[str, torch.Tensor]    # name -> (2, E) edge_index
    num_nodes: int
    multilabel: bool = False                   # y 가 (N, C) 멀티라벨이면 True (IMDB 등)


def _edge_adj(data, etype) -> sp.csr_matrix:
    """(src,rel,dst) sparse 인접. 그 방향이 없고 역방향(dst,rel,src)만 저장돼 있으면
    저장된 것의 전치를 쓴다(Freebase 처럼 단방향 저장 데이터셋 지원)."""
    src, rel, dst = etype
    etype = tuple(etype)
    if etype in data.edge_types:
        ei = data[etype].edge_index.numpy()
        n_src, n_dst = int(data[src].num_nodes), int(data[dst].num_nodes)
        return sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n_src, n_dst))
    rev = (dst, rel, src)
    if rev in data.edge_types:
        ei = data[rev].edge_index.numpy()
        n_d, n_s = int(data[dst].num_nodes), int(data[src].num_nodes)
        return sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n_d, n_s)).T.tocsr()
    raise KeyError(f"edge type {etype} (or reverse {rev}) not in {list(data.edge_types)}")


def _compose(data, edge_triples) -> sp.csr_matrix:
    """edge_triples (list of [src,rel,dst]) 의 인접행렬을 순서대로 곱 -> 타겟x타겟."""
    W = _edge_adj(data, tuple(edge_triples[0]))
    for et in edge_triples[1:]:
        W = W @ _edge_adj(data, tuple(et))
    return W.tocsr()


def _knn_sparsify_edges(W: sp.csr_matrix, k: int) -> torch.Tensor:
    """가중 인접 W 에서 행마다 상위 k 이웃만 남긴 무방향 edge_index (2,E)."""
    W = W.tocsr().copy()
    W.setdiag(0)
    W.eliminate_zeros()
    rows, cols = [], []
    n = W.shape[0]
    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        idx = W.indices[start:end]
        val = W.data[start:end]
        if len(idx) == 0:
            continue
        if len(idx) > k:
            top = np.argpartition(val, -k)[-k:]
            idx = idx[top]
        rows.extend([i] * len(idx))
        cols.extend(idx.tolist())
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long)
    e = np.array([rows, cols], dtype=np.int64)
    e = np.concatenate([e, e[[1, 0]]], axis=1)  # 무방향 대칭
    e = np.unique(e, axis=1)
    return torch.from_numpy(e)


def build_bundle(name: str, config: dict, data_root: str = "./data") -> HeteroBundle:
    """레지스트리 로더 + config 의 관계 정의로 HeteroBundle 구성."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. available: {list_datasets()}")
    data, target = DATASETS[name](data_root)
    tgt = data[target]
    n_full = int(tgt.num_nodes)
    masks_full = {k: getattr(tgt, f"{k}_mask").bool() for k in ("train", "val", "test")}
    multilabel = tgt.y.dim() > 1  # (N, C) 멀티라벨 (IMDB 등)
    if multilabel:
        y_full = tgt.y.float()
        num_classes = int(y_full.shape[1])
    else:
        y_full = tgt.y.long()
        num_classes = int(y_full[y_full >= 0].max().item()) + 1

    knn_k = config.get("pdgnn", {}).get("knn_k", 20)
    max_n = config.get("max_target_nodes", None)

    # 기저 관계를 sparse 로 합성(타겟×타겟). 큰 그래프도 dense 화 전에 subsample 하므로 안전.
    sparse_rels = {rn: _compose(data, tr) for rn, tr in config["base_relations"].items()}

    # dense GTN(N×N bmm) / HKS(eigh O(N^3)) 제약 → 너무 크면 연결 부분그래프로 subsample.
    if max_n is not None and n_full > max_n:
        union = sparse_rels[next(iter(sparse_rels))].copy()
        for W in list(sparse_rels.values())[1:]:
            union = union + W
        kept = _connected_subsample(union, masks_full["train"].numpy(), max_n,
                                    seed=config.get("seed", 0))
    else:
        kept = np.arange(n_full)
    kept_t = torch.from_numpy(kept).long()
    n = len(kept)

    y = y_full[kept_t]
    masks = {k: v[kept_t] for k, v in masks_full.items()}
    # 노드 특징 없으면 one-hot 항등행렬(= HAN/GTN 의 입력 Linear 가 임베딩 역할).
    x = torch.eye(n, dtype=torch.float32) if getattr(tgt, "x", None) is None else tgt.x.float()[kept_t]

    base_relations: Dict[str, torch.Tensor] = {}
    for rel_name, W in sparse_rels.items():
        Wk = W[kept][:, kept]                          # 부분그래프로 제한
        ei = _knn_sparsify_edges(Wk, knn_k)            # GTN 입력 기저관계도 kNN 희소화(필수)
        dense = torch.zeros((n, n), dtype=torch.float32)
        if ei.numel():
            dense[ei[0], ei[1]] = 1.0
        base_relations[rel_name] = dense

    han_metapaths: Dict[str, torch.Tensor] = {}
    for rel_name in config["han_metapaths"]:
        han_metapaths[rel_name] = _knn_sparsify_edges(sparse_rels[rel_name][kept][:, kept], knn_k)

    return HeteroBundle(
        name=name, target=target, x=x, y=y, masks=masks, num_classes=num_classes,
        base_relations=base_relations, han_metapaths=han_metapaths, num_nodes=n,
        multilabel=multilabel,
    )


def _connected_subsample(union, train_mask, cap: int, seed: int = 0) -> np.ndarray:
    """union sparse 인접(타겟×타겟)에서 train 노드 시작 BFS 로 ~cap 개 연결 타겟 노드 선택.
    정렬된 노드 인덱스 배열 반환. networkx 없이 CSR 직접 BFS(대형 그래프 메모리 안전)."""
    from collections import deque

    W = sp.csr_matrix(union)
    W.setdiag(0)
    W.eliminate_zeros()
    rng = np.random.RandomState(seed)
    train_idx = np.where(train_mask[: W.shape[0]])[0]
    start = int(rng.choice(train_idx)) if len(train_idx) else 0
    seen = {start}
    order = [start]
    q = deque([start])
    while q and len(order) < cap:
        u = q.popleft()
        for v in W.indices[W.indptr[u]:W.indptr[u + 1]]:
            v = int(v)
            if v not in seen:
                seen.add(v)
                order.append(v)
                q.append(v)
                if len(order) >= cap:
                    break
    return np.array(sorted(order[:cap]))


def get_dataset(name: str, config: dict, data_root: str = "./data") -> HeteroBundle:
    return build_bundle(name, config, data_root)
