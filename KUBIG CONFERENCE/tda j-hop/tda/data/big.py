"""대형 이종그래프 로더 (이종-그래프 subsample 사용): AMiner 등.

중간 노드(논문)가 수백만이라 전체 합성이 불가 → 라벨된 타겟 시드 중심으로 부분 이종그래프를
추출해 반환한다. 노드특징이 없으면 registry 에서 one-hot 으로 처리된다.
"""
from __future__ import annotations

from pathlib import Path

import torch

from tda.data._subsample import labels_from_y_index, subsample_hetero_graph, synth_splits


def load_aminer(data_root: str = "./data", cap: int = 4000):
    """AMiner (학술): target=author, 8 클래스, 노드특징 없음, 라벨 sparse(y_index)."""
    from torch_geometric.datasets import AMiner

    d = AMiner(root=f"{data_root}/AMiner")[0]
    target = "author"
    full_y = labels_from_y_index(d["author"].num_nodes, d["author"].y, d["author"].y_index)
    labeled_idx = d["author"].y_index.numpy()
    # subgraph 가 node 수와 안 맞는 속성(y_index, venue.y 등)을 건드리지 않도록 정리.
    for nt in list(d.node_types):
        for attr in ("y_index", "y"):
            if attr in d[nt]:
                del d[nt][attr]
    d["author"].y = full_y  # (num_nodes,) 전체 라벨(-1=미라벨)

    sub = subsample_hetero_graph(d, target, labeled_idx, cap=cap, hops=2, seed=0)
    labeled = (sub["author"].y >= 0).numpy()
    for k, m in synth_splits(labeled, seed=0).items():
        sub["author"][f"{k}_mask"] = torch.from_numpy(m)
    return sub, target


def load_entities(name: str, data_root: str = "./data", top_k: int = 6, cap: int = 6000):
    """RDF Entities (AIFB/MUTAG/BGS/AM): 단일 노드타입 'entity' + 다관계. R-GCN 고전 이종 NC.
    상위 top_k 빈도 관계만 edge type 으로 만들고(나머지는 무시), 노드특징 없음→one-hot.
    대형(BGS/AM)은 라벨 entity 중심으로 이종 subsample. val 은 train 에서 합성."""
    import numpy as np
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric.datasets import Entities

    d = Entities(root=f"{data_root}/Entities_{name}", name=name)[0]
    n = int(d.num_nodes)
    y = torch.full((n,), -1, dtype=torch.long)
    y[d.train_idx] = d.train_y
    y[d.test_idx] = d.test_y
    counts = torch.bincount(d.edge_type)
    topk = counts.argsort(descending=True)[:top_k].tolist()

    hd = HeteroData()
    hd["entity"].num_nodes = n
    hd["entity"].y = y
    tr = torch.zeros(n, dtype=torch.bool); tr[d.train_idx] = True
    te = torch.zeros(n, dtype=torch.bool); te[d.test_idx] = True
    hd["entity"].train_mask = tr
    hd["entity"].test_mask = te
    for j, r in enumerate(topk):
        ei = d.edge_index[:, d.edge_type == r]
        hd["entity", f"r{j}", "entity"].edge_index = ei

    if n > cap:
        labeled = (y >= 0).nonzero(as_tuple=False).squeeze(1).numpy()
        hd = subsample_hetero_graph(hd, "entity", labeled, cap=cap, hops=2, seed=0)
    # val 합성: 남은 train 의 15%
    p = hd["entity"]
    trm = p.train_mask.numpy().copy()
    idx = np.where(trm)[0]
    rng = np.random.RandomState(0); rng.shuffle(idx)
    cut = idx[int(0.85 * len(idx)):]
    val = np.zeros_like(trm); val[cut] = True; trm[cut] = False
    p.train_mask = torch.from_numpy(trm); p.val_mask = torch.from_numpy(val)
    return hd, "entity"


# cap=1000: 라벨 entity 전부 시드로 유지 + 중간 entity 를 ~4000(=cap*4) 까지만 채워 dense
# GTN/HKS 가 가능한 크기로. (RDF 는 라벨이 sparse 라 최종 노드수를 작게 잡아야 함.)
def load_aifb(data_root: str = "./data"):
    return load_entities("AIFB", data_root, top_k=6, cap=1000)


def load_mutag(data_root: str = "./data"):
    return load_entities("MUTAG", data_root, top_k=6, cap=1000)


def load_bgs(data_root: str = "./data"):
    return load_entities("BGS", data_root, top_k=6, cap=1000)


def load_am(data_root: str = "./data"):
    return load_entities("AM", data_root, top_k=6, cap=1500)


def load_hne(name: str, data_root: str = "./data", cap: int = 6000):
    """HNE 벤치마크 (yangji9181/HNE) 이종 그래프: PubMed(생의학)·Yelp(business) 등.
    포맷: node.dat(id,name,type,attrs) / link.dat(src,dst,link_type,w) / label.dat(+.test).
    노드타입은 t{int}, 관계는 r{int}. 라벨된 타입이 target. 특징 없으면 one-hot(registry).
    Yelp 는 멀티라벨. 대형은 라벨 노드 중심 이종 subsample."""
    import numpy as np
    import torch
    from torch_geometric.data import HeteroData

    base = Path(data_root) / "HNE" / name
    required = ["node.dat", "link.dat", "label.dat", "label.dat.test"]
    missing = [fname for fname in required if not (base / fname).is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"HNE {name} raw files are missing: {missing_list}. "
            f"Expected directory: {base}. "
            "Place the HNE-format files node.dat, link.dat, label.dat, and "
            "label.dat.test in that directory before running this dataset."
        )
    per_type, attrs = {}, {}
    with open(base / "node.dat") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            gid, nt = int(p[0]), int(p[2])
            per_type.setdefault(nt, []).append(gid)
            if len(p) > 3 and p[3]:
                attrs[gid] = np.fromstring(p[3], sep=",", dtype=np.float32)
    local = {}
    for t, ids in per_type.items():
        for i, g in enumerate(ids):
            local[g] = (t, i)

    hd = HeteroData()
    for t, ids in per_type.items():
        tn = f"t{t}"
        feats = [attrs.get(g) for g in ids]
        if all(fe is not None for fe in feats) and feats:
            hd[tn].x = torch.from_numpy(np.stack(feats))
        else:
            hd[tn].num_nodes = len(ids)

    edges = {}
    with open(base / "link.dat") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            s, d, lt = int(p[0]), int(p[1]), int(p[2])
            st, si = local[s]; dt, di = local[d]
            key = (f"t{st}", f"r{lt}", f"t{dt}")
            edges.setdefault(key, ([], []))
            edges[key][0].append(si); edges[key][1].append(di)
    for key, (ss, dd) in edges.items():
        hd[key].edge_index = torch.tensor([ss, dd], dtype=torch.long)

    # 라벨 파싱 (label.dat = train, label.dat.test = test). Yelp 멀티라벨.
    def read_labels(path):
        rows = []
        with open(path) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                rows.append((int(p[0]), int(p[2]), p[3]))
        return rows
    tr_rows = read_labels(base / "label.dat")
    te_rows = read_labels(base / "label.dat.test")
    ltype = tr_rows[0][1]
    tn = f"t{ltype}"
    n = len(per_type[ltype])
    multilabel = any("," in r[2] for r in tr_rows + te_rows)
    if multilabel:
        nc = 1 + max(int(x) for r in (tr_rows + te_rows) for x in r[2].split(","))
        y = torch.zeros((n, nc), dtype=torch.float32)
        for gid, _, lab in tr_rows + te_rows:
            for x in lab.split(","):
                y[local[gid][1], int(x)] = 1.0
    else:
        nc = 1 + max(int(r[2]) for r in tr_rows + te_rows)
        y = torch.full((n,), -1, dtype=torch.long)
        for gid, _, lab in tr_rows + te_rows:
            y[local[gid][1]] = int(lab)
    hd[tn].y = y
    trm = torch.zeros(n, dtype=torch.bool); tem = torch.zeros(n, dtype=torch.bool)
    for gid, _, _ in tr_rows:
        trm[local[gid][1]] = True
    for gid, _, _ in te_rows:
        tem[local[gid][1]] = True
    hd[tn].train_mask = trm; hd[tn].test_mask = tem

    if n > cap:
        labeled = (trm | tem).nonzero(as_tuple=False).squeeze(1).numpy()
        hd = subsample_hetero_graph(hd, tn, labeled, cap=cap, hops=2, seed=0)
    # val 합성
    p = hd[tn]
    a = p.train_mask.numpy().copy(); idx = np.where(a)[0]
    rng = np.random.RandomState(0); rng.shuffle(idx); cut = idx[int(0.85 * len(idx)):]
    v = np.zeros_like(a); v[cut] = True; a[cut] = False
    p.train_mask = torch.from_numpy(a); p.val_mask = torch.from_numpy(v)
    return hd, tn


def load_pubmed_hne(data_root: str = "./data"):
    return load_hne("PubMed", data_root, cap=6000)


def load_yelp_hne(data_root: str = "./data"):
    return load_hne("Yelp", data_root, cap=6000)


def load_mag(data_root: str = "./data", cap: int = 3000):
    """ogbn-mag (학술, 대형): target=paper, 349 클래스, paper 특징 128 + 공식 splits 보유."""
    from torch_geometric.datasets import OGB_MAG

    import numpy as np

    d = OGB_MAG(root=f"{data_root}/OGB_MAG")[0]
    target = "paper"
    p = d["paper"]
    # train/val/test 에서 균형 있게 시드 (train 만/proportional 이면 val·test 가 거의 안 남음).
    rng = np.random.RandomState(0)
    seeds = []
    for k, frac in (("train_mask", 0.5), ("val_mask", 0.25), ("test_mask", 0.25)):
        idx = getattr(p, k).nonzero(as_tuple=False).squeeze(1).numpy()
        seeds.append(rng.choice(idx, size=min(len(idx), int(cap * frac)), replace=False))
    labeled = np.concatenate(seeds)
    # 공식 마스크는 subgraph 가 노드 속성으로 함께 재인덱싱하므로 별도 synth 불필요.
    sub = subsample_hetero_graph(d, target, labeled, cap=cap, hops=2, seed=0)
    return sub, target
