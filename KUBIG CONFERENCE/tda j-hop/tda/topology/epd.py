"""Stage 2 — GTN 이 발견한 채널(메타패스) 그래프 위에서 PDGNN 위상 특징 추출.

채널마다:
  1. soft 인접행렬 H_c 를 kNN 으로 희소화 -> 이산 그래프 (PDGNN 은 이산 그래프 필요)
  2. HKS(K 스케일) 노드 필터 계산
  3. 노드 ego 그래프의 정확 EPD(gudhi lower-star)를 라벨로 PDGNN 학습
     (정확 EPD 는 **학습 라벨로만** — 추론 시 정확 계산 없음)
  4. 학습된 PDGNN 으로 노드별 예측 EPD -> persistence image (N, 25*K)

원본 TLC-GNN `hetero/pdgnn_metapath.py` 의 워크플로를 본 패키지의 PDGNN/HKS/PI 로
재구성한 것이다.
"""
from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np
import torch

from tda.models.pdgnn import PDGNN, bipartite_loss
from tda.topology.hks import compute_hks
from tda.topology.persistence_image import PersistenceImage


def knn_graph_from_dense(adj: torch.Tensor, k: int) -> nx.Graph:
    """soft (N,N) 인접에서 행마다 상위 k 이웃만 남긴 무방향 이산 그래프."""
    a = adj.detach().cpu().numpy()
    np.fill_diagonal(a, 0.0)
    n = a.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        row = a[i]
        nz = np.where(row > 0)[0]
        if len(nz) == 0:
            continue
        if len(nz) > k:
            nz = nz[np.argpartition(row[nz], -k)[-k:]]
        for j in nz:
            g.add_edge(i, int(j))
    return g


def _edge_index_of(g: nx.Graph) -> torch.Tensor:
    if g.number_of_edges() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    e = np.array(list(g.edges()), dtype=np.int64).T
    e = np.concatenate([e, e[[1, 0]]], axis=1)
    return torch.from_numpy(e)


def _ego_filt_edges(g: nx.Graph, center: int, hop: int, node_filt: dict, max_nodes: int):
    """ego 그래프의 (filt (m,), edge_index (2,E), node_list) — 지역 재인덱싱."""
    ego = nx.ego_graph(g, center, radius=hop)
    if ego.number_of_nodes() > max_nodes:
        keep = sorted(ego.nodes(), key=lambda nd: node_filt.get(nd, 0.0))[:max_nodes]
        ego = ego.subgraph(keep).copy()
    nodes = list(ego.nodes())
    if len(nodes) == 0:
        return None
    remap = {nd: i for i, nd in enumerate(nodes)}
    filt = np.array([node_filt.get(nd, 0.0) for nd in nodes], dtype=np.float64)
    if ego.number_of_edges() == 0:
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        e = np.array([(remap[u], remap[v]) for u, v in ego.edges()], dtype=np.int64).T
        ei = np.concatenate([e, e[[1, 0]]], axis=1)
    return filt, ei, nodes


def _diagram_points(st, max_filt: float):
    """SimplexTree 의 유한 (birth, death) 점 (H0+H1). inf death 는 max_filt 로 cap."""
    pts = []
    for _dim, (b, d) in st.persistence(homology_coeff_field=2, min_persistence=0.0):
        if d == float("inf"):
            d = max_filt
        if d > b:
            pts.append((b, d))
    return pts


def _exact_epd(filt: np.ndarray, ei: np.ndarray) -> np.ndarray:
    """ego 그래프의 정확 lower-star EPD (학습 라벨)."""
    import gudhi

    st = gudhi.SimplexTree()
    for i, f in enumerate(filt):
        st.insert([int(i)], filtration=float(f))
    seen = set()
    for a, b in zip(ei[0], ei[1]):
        e = (int(min(a, b)), int(max(a, b)))
        if e in seen:
            continue
        seen.add(e)
        st.insert([e[0], e[1]], filtration=float(max(filt[e[0]], filt[e[1]])))
    max_filt = float(filt.max()) if filt.size else 1.0
    return np.array(_diagram_points(st, max_filt), dtype=np.float64)


def gen_training_samples(g: nx.Graph, hks: np.ndarray, hop: int, max_nodes: int,
                         n_samples: int, seed: int = 0):
    """(node, scale) ego 들에서 (filt(m,1), ei(2,E), gt_epd(P,2)) 학습 샘플 생성."""
    rng = np.random.RandomState(seed)
    n, K = hks.shape
    nodes = rng.choice(n, size=min(n_samples, n), replace=False)
    samples = []
    for v in nodes:
        for k in range(K):
            node_filt = {nd: float(hks[nd, k]) for nd in g.nodes()}
            res = _ego_filt_edges(g, int(v), hop, node_filt, max_nodes)
            if res is None:
                continue
            filt, ei, _ = res
            if ei.shape[1] == 0:
                continue
            gt = _exact_epd(filt, ei)
            if gt.size == 0:
                continue
            samples.append((filt.reshape(-1, 1).astype(np.float32),
                            ei.astype(np.int64), gt.astype(np.float32)))
    return samples


def train_pdgnn(samples, hidden=32, layers=3, epochs=30, lr=1e-3, seed=1234,
                device=None, verbose=False, agg="sum_min") -> PDGNN:
    """ego EPD 라벨로 PDGNN 학습 (bipartite loss). agg='sum_min'(원본)|'sum'(D2 ablation)."""
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PDGNN(hidden_dim=hidden, num_layers=layers, agg=agg).to(dev)
    if not samples:
        return model
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(samples))
        losses = []
        for idx in order:
            filt, ei, gt = samples[idx]
            ft = torch.tensor(filt, device=dev)
            et = torch.tensor(ei, device=dev)
            gtt = torch.tensor(gt, device=dev)
            opt.zero_grad()
            loss = bipartite_loss(model(ft, et), gtt)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"      [pdgnn] ep {ep:3d} loss={np.mean(losses) if losses else 0:.4f}", flush=True)
    return model


@torch.no_grad()
def predict_node_pi(model: PDGNN, g: nx.Graph, hks: np.ndarray, hop: int,
                    max_nodes: int, imager: PersistenceImage, device=None) -> np.ndarray:
    """(N, res^2 * K) PDGNN 예측 EPD -> 노드별 persistence image (정확 계산 없음)."""
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    n, K = hks.shape
    R2 = imager.dim
    out = np.zeros((n, R2 * K), dtype=np.float64)
    filts_by_k = [{nd: float(hks[nd, k]) for nd in g.nodes()} for k in range(K)]
    for v in range(n):
        for k in range(K):
            res = _ego_filt_edges(g, v, hop, filts_by_k[k], max_nodes)
            if res is None:
                continue
            filt, ei, _ = res
            if ei.shape[1] == 0:
                continue
            ft = torch.tensor(filt.reshape(-1, 1).astype(np.float32), device=dev)
            et = torch.tensor(ei.astype(np.int64), device=dev)
            pred = model(ft, et).cpu().numpy()
            pred = pred[pred[:, 1] > pred[:, 0]]
            if pred.size:
                out[v, k * R2:(k + 1) * R2] = imager.transform(pred)
    return out


def compute_channel_topology(adj: torch.Tensor, config: dict, seed: int,
                             device=None, verbose: bool = False) -> np.ndarray:
    """한 채널 인접행렬 -> (N, res^2 * K) PDGNN 위상 특징."""
    pc = config["pdgnn"]
    # stage2(PDGNN 입력) 전용 kNN. 기본은 knn_k 와 동일하되, D1(no-kNN) ablation 에서
    # base-relation/HAN kNN(=knn_k)은 그대로 두고 이 값만 키워 PDGNN 앞 희소화만 제거한다.
    g = knn_graph_from_dense(adj, pc.get("channel_knn_k", pc["knn_k"]))
    ei = _edge_index_of(g)
    hks = compute_hks(ei, num_nodes=adj.size(0), K=pc["hks_K"], device=device, verbose=verbose)
    samples = gen_training_samples(g, hks, pc["hop"], pc["max_nodes"], pc["n_train_samples"], seed)
    model = train_pdgnn(samples, hidden=pc["hidden_dim"], layers=pc["layers"],
                        epochs=pc["epochs"], lr=pc["lr"], seed=seed + 1234,
                        device=device, verbose=verbose, agg=pc.get("agg", "sum_min"))
    # PI 범위/σ 는 데이터셋마다 튜닝 가능(유연). HKS 가 z-정규화돼 기본값은 대체로 통용되지만
    # config 로 덮어쓸 수 있다. 미지정 시 z-norm HKS 에 맞춘 기본값 사용.
    imager = PersistenceImage(
        resolution=pc["pi_resolution"],
        sigma=pc.get("pi_sigma", 0.5),
        birth_range=tuple(pc.get("pi_birth_range", (-3.0, 3.0))),
        pers_range=tuple(pc.get("pi_pers_range", (0.0, 6.0))),
    )
    return predict_node_pi(model, g, hks, pc["hop"], pc["max_nodes"], imager, device=device)
