"""Convert homogeneous molecular graphs to exact-atom-type HeteroData."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from ogb.utils.features import allowable_features
from torch_geometric.data import Data, HeteroData

_RAW_ATOMIC_ENTRIES = list(allowable_features["possible_atomic_num_list"])
BOND_TYPE_LIST: List[str] = list(allowable_features["possible_bond_type_list"])


def _atomic_z_from_feature_index(idx: int) -> int:
    """Map OGB atom feature column-0 index to atomic number Z."""
    if idx < 0 or idx >= len(_RAW_ATOMIC_ENTRIES):
        raise ValueError(f"Invalid atomic number index {idx}")
    val = _RAW_ATOMIC_ENTRIES[idx]
    if isinstance(val, str):
        if val.isdigit():
            return int(val)
        # OGB uses 'misc' for out-of-vocabulary atoms.
        return 6
    return int(val)


ATOMIC_NUM_LIST: List[int] = [_atomic_z_from_feature_index(i) for i in range(len(_RAW_ATOMIC_ENTRIES))]

# Index → atomic number lookup (CPU); safe for DataLoader workers.
_ATOMIC_IDX_TO_Z = torch.tensor(ATOMIC_NUM_LIST, dtype=torch.long)

# Approximate mass number lookup (integer amu) for optional node typing.
ATOMIC_MASS_LOOKUP: Dict[int, int] = {
    1: 1,
    6: 12,
    7: 14,
    8: 16,
    9: 19,
    15: 31,
    16: 32,
    17: 35,
    35: 79,
    53: 127,
}


def decode_atomic_number(atom_feature_row: torch.Tensor) -> int:
    """Decode OGB categorical atom feature column 0 to atomic number Z."""
    idx = int(atom_feature_row[0].item())
    return _atomic_z_from_feature_index(idx)


def atomic_number_to_mass_number(z: int) -> int:
    if z in ATOMIC_MASS_LOOKUP:
        return ATOMIC_MASS_LOOKUP[z]
    # Fallback: coarse placeholder when exact mass is not in the lookup table.
    return z if z <= 20 else int(round(z * 1.6))


def atom_to_node_type(atom_feature_row: torch.Tensor, mode: str = "atomic_number") -> str:
    z = decode_atomic_number(atom_feature_row)
    if mode == "atomic_mass":
        mass = atomic_number_to_mass_number(z)
        return f"atom_mass_{mass}"
    return f"atom_Z_{z}"


def bond_to_edge_type(edge_attr_row: torch.Tensor) -> str:
    idx = int(edge_attr_row[0].item())
    if idx < 0 or idx >= len(BOND_TYPE_LIST):
        return "bond_unknown"
    name = BOND_TYPE_LIST[idx].lower()
    if name == "misc":
        return "bond_unknown"
    return f"bond_{name}"


def compute_atomic_numbers(data: Data) -> torch.Tensor:
    idx = data.x[:, 0].long().clamp(0, len(ATOMIC_NUM_LIST) - 1)
    return _ATOMIC_IDX_TO_Z.to(idx.device)[idx]


def homo_to_hetero(
    data: Data,
    node_type_mode: str = "atomic_number",
) -> Tuple[HeteroData, Dict[str, torch.Tensor]]:
    """
    Convert PyG molecular Data to HeteroData with exact atom node types and bond edge types.

    Returns:
        hetero: HeteroData with typed nodes/edges
        aux: dict with atomic_numbers [N], node_types [N] strings as list metadata
    """
    num_nodes = int(data.num_nodes)
    atomic_numbers = compute_atomic_numbers(data)
    node_types = [atom_to_node_type(data.x[i], mode=node_type_mode) for i in range(num_nodes)]

    hetero = HeteroData()
    hetero.y = data.y
    if hasattr(data, "graph_idx"):
        hetero.graph_idx = data.graph_idx

    # Group node features by type; local_index[i] = position of node i within its type bucket.
    type_to_indices: Dict[str, List[int]] = {}
    local_index = torch.zeros(num_nodes, dtype=torch.long)
    for i, ntype in enumerate(node_types):
        bucket = type_to_indices.setdefault(ntype, [])
        local_index[i] = len(bucket)
        bucket.append(i)

    for ntype, global_indices in type_to_indices.items():
        idx = torch.tensor(global_indices, dtype=torch.long)
        hetero[ntype].x = data.x[idx]
        hetero[ntype].global_index = idx  # maps local -> original homogeneous index

    edge_index = data.edge_index
    edge_attr = data.edge_attr
    edge_buckets: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}

    for e in range(edge_index.size(1)):
        u = int(edge_index[0, e].item())
        v = int(edge_index[1, e].item())
        src_type = node_types[u]
        dst_type = node_types[v]
        bond_type = bond_to_edge_type(edge_attr[e])
        rel = (src_type, bond_type, dst_type)
        edge_buckets.setdefault(rel, []).append((local_index[u].item(), local_index[v].item()))

    for rel, pairs in edge_buckets.items():
        src_t, bond_t, dst_t = rel
        if not pairs:
            continue
        src_loc, dst_loc = zip(*pairs)
        hetero[src_t, bond_t, dst_t].edge_index = torch.tensor(
            [list(src_loc), list(dst_loc)], dtype=torch.long
        )

    aux = {
        "atomic_numbers": atomic_numbers,
        "node_types": node_types,
        "local_index": local_index,
        "num_nodes": torch.tensor([num_nodes], dtype=torch.long),
    }
    return hetero, aux


def hetero_node_embeddings_to_homo(
    typed_embeddings: Dict[str, torch.Tensor],
    hetero: HeteroData,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Scatter typed node embeddings back to original homogeneous node order."""
    hidden_dim = next(iter(typed_embeddings.values())).size(-1)
    out = torch.zeros(num_nodes, hidden_dim, device=device)
    for ntype, emb in typed_embeddings.items():
        if not hasattr(hetero[ntype], "global_index"):
            continue
        global_idx = hetero[ntype].global_index.to(device)
        out[global_idx] = emb
    return out


def scatter_batched_hetero_to_homo(
    typed_embeddings: Dict[str, torch.Tensor],
    batch_hetero: HeteroData,
    ptr: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Map batched HeteroData node embeddings back to PyG Batch node order."""
    hidden_dim = next(iter(typed_embeddings.values())).size(-1)
    out = torch.zeros(num_nodes, hidden_dim, device=device)
    ptr = ptr.to(device)
    for ntype, emb in typed_embeddings.items():
        store = batch_hetero[ntype]
        if not hasattr(store, "global_index"):
            continue
        g_ids = store.batch.to(device)
        local_idx = store.global_index.to(device)
        global_idx = ptr[g_ids] + local_idx
        out[global_idx] = emb
    return out
