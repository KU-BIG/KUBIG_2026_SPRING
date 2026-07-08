"""Graph-topology (hop-distance) Vietoris-Rips filtration TDA features.

Two lenses:
  A: full molecular graph, hop distance (bonds counted as 1 step each)
     -> H1 captures ring systems at their intrinsic graph scale (ring size).
  B: aromatic-bond subgraph only, hop distance
     -> H1 isolates aromatic ring systems, ignoring the rest of the scaffold.

Both use only connectivity (no 3D coordinates, no chemistry beyond bond type),
so they are trivially rotation/translation invariant.
"""
from __future__ import annotations

from typing import Optional

import gudhi
import networkx as nx
import numpy as np
import torch
from ogb.utils.features import allowable_features
from torch_geometric.data import Data

from config import GRAPH_RIPS_DIM, PI_RESOLUTION, PI_SIGMA, RIPS_HOP_MAX
from features.tda_utils import PersistencePoint, persistence_image

_BOND_TYPE_LIST = [str(b).upper() for b in allowable_features["possible_bond_type_list"]]
AROMATIC_BOND_IDX = _BOND_TYPE_LIST.index("AROMATIC") if "AROMATIC" in _BOND_TYPE_LIST else 3


def graph_to_hop_distance_matrix(data: Data, aromatic_only: bool = False) -> np.ndarray:
    """All-pairs shortest-path (hop) distance matrix. Disconnected pairs = +inf."""
    num_nodes = int(data.num_nodes)
    ei = data.edge_index.cpu().numpy()
    ea = data.edge_attr.cpu().numpy()

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if aromatic_only and int(ea[e, 0]) != AROMATIC_BOND_IDX:
            continue
        g.add_edge(u, v)

    dist = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for src, lengths in nx.all_pairs_shortest_path_length(g):
        for dst, d in lengths.items():
            dist[src, dst] = float(d)
    return dist


def _persistence_from_distance_matrix(
    dist: np.ndarray,
    max_edge_length: float = RIPS_HOP_MAX,
) -> list[list[PersistencePoint]]:
    """H0/H1 persistence from a Rips complex on a (hop) distance matrix."""
    # Keep disconnected pairs above the cap so they never link within the filtration.
    dist = np.where(np.isinf(dist), max_edge_length + 10.0, dist)
    rips = gudhi.RipsComplex(distance_matrix=dist, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()

    diagrams: list[list[PersistencePoint]] = [[], []]  # H0, H1
    for dim, (birth, death) in st.persistence():
        if dim > 1:
            continue
        if birth == float("inf") or death == float("inf"):
            continue
        if death <= birth:
            continue
        diagrams[dim].append(PersistencePoint(float(birth), float(death)))
    return diagrams


def compute_graph_rips_vector(
    data: Data,
    aromatic_only: bool = False,
    resolution: int = PI_RESOLUTION,
    sigma: float = PI_SIGMA,
) -> np.ndarray:
    """Persistence-image vector (H0 + H1) for a hop-distance Rips filtration."""
    if data.edge_index.numel() == 0:
        return np.zeros(GRAPH_RIPS_DIM, dtype=np.float32)

    dist = graph_to_hop_distance_matrix(data, aromatic_only=aromatic_only)
    diagrams = _persistence_from_distance_matrix(dist)

    vectors = [
        persistence_image(
            diag,
            resolution=resolution,
            sigma=sigma,
            birth_range=(0.0, RIPS_HOP_MAX),
            pers_range=(0.0, RIPS_HOP_MAX),
        )
        for diag in diagrams
    ]
    vector = np.concatenate(vectors, axis=0).astype(np.float32)

    if vector.shape[0] < GRAPH_RIPS_DIM:
        vector = np.pad(vector, (0, GRAPH_RIPS_DIM - vector.shape[0]))
    elif vector.shape[0] > GRAPH_RIPS_DIM:
        vector = vector[:GRAPH_RIPS_DIM]
    return vector


def compute_graph_rips_batch(
    dataset,
    aromatic_only: bool = False,
    indices: Optional[list[int]] = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Compute hop-distance Rips vectors for selected graphs."""
    if indices is None:
        indices = list(range(len(dataset)))

    iterator = indices
    if show_progress:
        from tqdm import tqdm

        desc = "Aromatic Rips TDA" if aromatic_only else "Graph Rips TDA"
        iterator = tqdm(indices, desc=desc)

    features = [compute_graph_rips_vector(dataset[idx], aromatic_only=aromatic_only) for idx in iterator]
    return torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
