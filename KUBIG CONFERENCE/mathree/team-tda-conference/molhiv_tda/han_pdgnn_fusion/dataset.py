"""OGBG-MolHIV dataset loading for HAN+PDGNN fusion models."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.loader import DataLoader

# Reuse parent project's OGB loader (handles torch.load compat).
_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from data.load_molhiv import load_molhiv, prepare_dataset  # noqa: E402
from hetero_transform import homo_to_hetero, load_or_build_hetero_cache  # noqa: E402


class MolHIVFusionDataset(Dataset):
    """Wrap OGB graphs; optionally attach precomputed hetero metadata per graph."""

    def __init__(
        self,
        base_dataset,
        indices: Sequence[int],
        node_type_mode: str = "atomic_number",
        build_hetero: bool = False,
        hetero_cache: Optional[Dict[int, HeteroData]] = None,
    ):
        self.base_dataset = base_dataset
        self.indices = list(map(int, indices))
        self.node_type_mode = node_type_mode
        self.build_hetero = build_hetero
        self.hetero_cache = hetero_cache

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Data:
        graph_idx = self.indices[pos]
        data = self.base_dataset[graph_idx].clone()
        data.graph_idx = torch.tensor([graph_idx], dtype=torch.long)
        if self.build_hetero:
            if self.hetero_cache is not None:
                data.hetero_data = self.hetero_cache[graph_idx]
            else:
                hetero, _ = homo_to_hetero(data, node_type_mode=self.node_type_mode)
                data.hetero_data = hetero
        return data


def collate_mol_batch(items: list[Data]) -> Data:
    """Batch homogeneous graphs; hetero built in model forward for flexibility."""
    return Batch.from_data_list(items)


def get_dataset(
    dataset_root: str | Path,
    node_type_mode: str = "atomic_number",
):
    root = Path(dataset_root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()
    if not (root / "ogbg_molhiv").exists():
        from download_molhiv import download_molhiv

        download_molhiv(root)
    prepare_dataset(root)
    dataset, split_idx, evaluator, smiles = load_molhiv(root)
    return dataset, split_idx, evaluator, smiles


def make_dataloaders(
    dataset,
    split_idx: Dict[str, torch.Tensor],
    batch_size: int = 32,
    num_workers: int = 0,
    node_type_mode: str = "atomic_number",
    max_samples: Optional[int] = None,
    cache_root: Optional[Path] = None,
) -> Dict[str, DataLoader]:
    hetero_cache = None
    if cache_root is not None:
        cache_path = Path(cache_root) / f"hetero_{node_type_mode}.pt"
        all_indices = sorted(
            {
                int(i)
                for split in ("train", "valid", "test")
                for i in split_idx[split].tolist()
            }
        )
        hetero_cache = load_or_build_hetero_cache(cache_path, dataset, all_indices, node_type_mode)

    loaders = {}
    for split in ("train", "valid", "test"):
        indices = split_idx[split].tolist()
        if max_samples is not None:
            indices = indices[:max_samples]
        subset = MolHIVFusionDataset(
            dataset,
            indices,
            node_type_mode=node_type_mode,
            build_hetero=hetero_cache is not None,
            hetero_cache=hetero_cache,
        )
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_mol_batch,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
