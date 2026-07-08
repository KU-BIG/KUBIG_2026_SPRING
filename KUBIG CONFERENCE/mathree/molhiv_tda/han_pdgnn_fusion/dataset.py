"""OGBG-MolHIV dataset loading for HAN+PDGNN fusion models."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

# Reuse parent project's OGB loader (handles torch.load compat).
_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from data.load_molhiv import load_molhiv, prepare_dataset  # noqa: E402
from hetero_transform import homo_to_hetero  # noqa: E402


class MolHIVFusionDataset(Dataset):
    """Wrap OGB graphs; optionally pre-build hetero metadata per graph."""

    def __init__(
        self,
        base_dataset,
        indices: Sequence[int],
        node_type_mode: str = "atomic_number",
        build_hetero: bool = True,
    ):
        self.base_dataset = base_dataset
        self.indices = list(map(int, indices))
        self.node_type_mode = node_type_mode
        self.build_hetero = build_hetero

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Data:
        graph_idx = self.indices[pos]
        data = self.base_dataset[graph_idx].clone()
        data.graph_idx = torch.tensor([graph_idx], dtype=torch.long)
        if self.build_hetero:
            hetero, aux = homo_to_hetero(data, node_type_mode=self.node_type_mode)
            data.hetero_data = hetero
            data.atomic_numbers = aux["atomic_numbers"]
        return data


def collate_mol_batch(items: list[Data]) -> Batch:
    """Batch homogeneous graphs; optionally attach pre-built hetero graphs for HAN."""
    batch = Batch.from_data_list(items)
    if items and hasattr(items[0], "hetero_data"):
        batch.hetero_list = [item.hetero_data for item in items]
    return batch


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
    from download_molhiv import ensure_smiles_mapping

    ensure_smiles_mapping(root)
    dataset, split_idx, evaluator, smiles = load_molhiv(root)
    return dataset, split_idx, evaluator, smiles


def make_dataloaders(
    dataset,
    split_idx: Dict[str, torch.Tensor],
    batch_size: int = 32,
    num_workers: int = 0,
    node_type_mode: str = "atomic_number",
    max_samples: Optional[int] = None,
    build_hetero: bool = True,
) -> Dict[str, DataLoader]:
    # HeteroData preload only in the main process; worker pickling exhausts FDs on Elice.
    preload_hetero = build_hetero and num_workers == 0
    loaders = {}
    for split in ("train", "valid", "test"):
        indices = split_idx[split].tolist()
        if max_samples is not None:
            indices = indices[:max_samples]
        subset = MolHIVFusionDataset(
            dataset,
            indices,
            node_type_mode=node_type_mode,
            build_hetero=preload_hetero,
        )
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_mol_batch,
            pin_memory=torch.cuda.is_available() and num_workers == 0,
        )
    return loaders
