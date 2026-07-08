"""Bond-type filtration TDA features for molecular graphs."""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from config import BOND_TDA_DIM, PI_RESOLUTION, PI_SIGMA
from features.tda_utils import diagrams_to_vector, gudhi_persistence_from_simplices


def pyg_to_bond_filtration_graph(data: Data) -> nx.Graph:
    """Build a NetworkX graph with bond-type edge filtration values."""
    edge_index = data.edge_index.cpu().numpy()
    bond_type = data.edge_attr[:, 0].cpu().numpy().astype(float)
    num_nodes = int(data.num_nodes)

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))

    for e_idx in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
        bt = float(bond_type[e_idx])
        if g.has_edge(u, v):
            g[u][v]["bond_type"] = min(g[u][v]["bond_type"], bt)
        else:
            g.add_edge(u, v, bond_type=bt)

    for node in g.nodes():
        incident = [g[node][nbr]["bond_type"] for nbr in g.neighbors(node)]
        g.nodes[node]["filtration"] = min(incident) if incident else 0.0

    return g


def graph_to_simplices(g: nx.Graph) -> list[tuple[list[int], float]]:
    """Convert bond filtration graph to simplex list for GUDHI."""
    simplices: list[tuple[list[int], float]] = []
    for node, attrs in g.nodes(data=True):
        simplices.append(([node], float(attrs["filtration"])))
    for u, v, attrs in g.edges(data=True):
        node_f = max(g.nodes[u]["filtration"], g.nodes[v]["filtration"])
        edge_f = max(float(attrs["bond_type"]), node_f)
        simplices.append(([u, v], edge_f))
    simplices.sort(key=lambda x: x[1])
    return simplices


def compute_bond_tda_vector(
    data: Data,
    resolution: int = PI_RESOLUTION,
    sigma: float = PI_SIGMA,
) -> np.ndarray:
    """Compute bond-type filtration persistence image for one molecule."""
    if data.edge_index.numel() == 0:
        return np.zeros(BOND_TDA_DIM, dtype=np.float32)

    g = pyg_to_bond_filtration_graph(data)
    simplices = graph_to_simplices(g)
    diagrams = gudhi_persistence_from_simplices(simplices, max_dim=1)
    vector = diagrams_to_vector(diagrams, resolution=resolution, sigma=sigma)

    if vector.shape[0] < BOND_TDA_DIM:
        vector = np.pad(vector, (0, BOND_TDA_DIM - vector.shape[0]))
    elif vector.shape[0] > BOND_TDA_DIM:
        vector = vector[:BOND_TDA_DIM]
    return vector.astype(np.float32)


def compute_bond_tda_batch(
    dataset,
    indices: Optional[list[int]] = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Compute bond-type TDA vectors for selected graphs."""
    if indices is None:
        indices = list(range(len(dataset)))

    features = []
    iterator = indices
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(indices, desc="Bond-type TDA")

    for idx in iterator:
        data = dataset[idx]
        features.append(compute_bond_tda_vector(data))

    return torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
