"""대형 이종그래프를 다루기 위한 이종-그래프 수준 subsampling.

AMiner/ogbn-mag/RCDD 처럼 중간 노드(논문 등)가 수백만이면 타겟×타겟 메타패스 합성이
불가능하다. 그래서 라벨된 타겟 노드 시드에서 이종 엣지를 따라 BFS 로 작은 연결 부분
이종그래프를 추출(`HeteroData.subgraph`)한 뒤, 그 위에서 build_bundle 이 합성하도록 한다.
"""
from __future__ import annotations

import numpy as np
import torch


def subsample_hetero_graph(data, target: str, labeled_idx: np.ndarray, cap: int,
                           hops: int = 2, seed: int = 0):
    """라벨된 target 노드 시드에서 hops-홉 이종 BFS 로 부분 이종그래프 추출.

    각 노드타입은 최대 cap*expand 개로 제한. 반환: 재인덱싱된 작은 HeteroData.
    target 노드는 시드(라벨된)만 유지하므로 build_bundle 의 마스크가 라벨 노드를 가리킨다.
    """
    rng = np.random.RandomState(seed)
    n_seed = min(len(labeled_idx), cap)
    seeds = rng.choice(labeled_idx, size=n_seed, replace=False)
    kept = {nt: set() for nt in data.node_types}
    kept[target] = set(int(s) for s in seeds)
    frontier = {nt: set() for nt in data.node_types}
    frontier[target] = set(kept[target])
    expand_cap = cap * 4  # 중간 노드타입 상한

    for _ in range(hops):
        new_frontier = {nt: set() for nt in data.node_types}
        for (s, r, dd) in data.edge_types:
            if not frontier[s]:
                continue
            ei = data[(s, r, dd)].edge_index
            src_t = torch.tensor(sorted(frontier[s]), dtype=torch.long)
            m = torch.isin(ei[0], src_t)
            dsts = ei[1][m].unique().tolist()
            room = expand_cap - len(kept[dd])
            if room <= 0:
                continue
            if len(dsts) > room:
                dsts = list(rng.choice(dsts, size=room, replace=False))
            fresh = [int(v) for v in dsts if int(v) not in kept[dd]]
            kept[dd].update(fresh)
            new_frontier[dd].update(fresh)
        frontier = new_frontier

    subset = {nt: torch.tensor(sorted(idx), dtype=torch.long)
              for nt, idx in kept.items() if len(idx) > 0}
    return data.subgraph(subset)


def labels_from_y_index(num_nodes: int, y: torch.Tensor, y_index: torch.Tensor):
    """sparse 라벨(y on y_index)을 (num_nodes,) 전체 벡터로(미라벨=-1) 펼친다."""
    full = torch.full((num_nodes,), -1, dtype=torch.long)
    full[y_index] = y.long()
    return full


def synth_splits(labeled_mask: np.ndarray, seed: int = 0, ratios=(0.6, 0.2)):
    """라벨된 노드를 train/val/test 로 무작위 분할(시드 고정). 반환: dict of bool 마스크."""
    idx = np.where(labeled_mask)[0]
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n = len(idx)
    a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
    masks = {}
    for key, sub in (("train", idx[:a]), ("val", idx[a:b]), ("test", idx[b:])):
        m = np.zeros(len(labeled_mask), dtype=bool)
        m[sub] = True
        masks[key] = m
    return masks
