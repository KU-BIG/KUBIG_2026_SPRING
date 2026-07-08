"""OGBG-MolHIV loading, SMILES mapping, and feature-aware datasets."""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import torch
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from config import DATASET_NAME, DATASET_ROOT, SMILES_CSV

_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    """PyTorch 2.6+ defaults weights_only=True; OGB processed files need False."""
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


def _patch_torch_load():
    torch.load = _torch_load_compat


def _restore_torch_load():
    torch.load = _TORCH_LOAD


def prepare_dataset(root: Path | str = DATASET_ROOT) -> Path:
    """Ensure OGB dataset directory is ready for current PyG."""
    root = Path(root)
    dataset_dir = root / "ogbg_molhiv"
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. "
            "Symlink or copy ogbg-molhiv data into dataset/ogbg_molhiv."
        )

    for sub in ("raw", "split/scaffold"):
        folder = dataset_dir / sub
        if not folder.exists():
            continue
        for csv_path in folder.glob("*.csv"):
            gz_path = csv_path.with_suffix(csv_path.suffix + ".gz")
            if not gz_path.exists():
                with open(csv_path, "rb") as fin, gzip.open(gz_path, "wb") as fout:
                    shutil.copyfileobj(fin, fout)

    processed = dataset_dir / "processed" / "geometric_data_processed.pt"
    if processed.exists():
        try:
            _patch_torch_load()
            try:
                torch.load(processed, weights_only=False)
            finally:
                _restore_torch_load()
        except Exception:
            shutil.rmtree(dataset_dir / "processed", ignore_errors=True)

    return dataset_dir


def load_smiles_mapping(csv_path: Path | str = SMILES_CSV) -> list[str]:
    """Return SMILES list where index i matches graph i in the dataset."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError(f"Expected 'smiles' column in {csv_path}")
    return df["smiles"].astype(str).tolist()


def load_molhiv(root: Path | str = DATASET_ROOT):
    """Load dataset, split indices, and evaluator."""
    prepare_dataset(root)
    _patch_torch_load()
    try:
        dataset = PygGraphPropPredDataset(name=DATASET_NAME, root=str(root))
        split_idx = dataset.get_idx_split()
    finally:
        _restore_torch_load()
    evaluator = Evaluator(name=DATASET_NAME)
    smiles = load_smiles_mapping()
    if len(smiles) != len(dataset):
        raise ValueError(
            f"SMILES count ({len(smiles)}) != dataset size ({len(dataset)})"
        )
    return dataset, split_idx, evaluator, smiles


class IndexedGraphDataset(torch.utils.data.Dataset):
    """Wrap OGB graphs with stable graph indices for feature lookup."""

    def __init__(self, dataset: PygGraphPropPredDataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(map(int, indices))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Data:
        graph_idx = self.indices[pos]
        data = self.dataset[graph_idx].clone()
        data.graph_idx = torch.tensor([graph_idx], dtype=torch.long)
        return data


def make_loaders(
    dataset: PygGraphPropPredDataset,
    split_idx: Dict[str, torch.Tensor],
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: int | None = None,
):
    """Create train/valid/test DataLoaders using official OGB split."""
    loaders = {}
    for split in ("train", "valid", "test"):
        indices = split_idx[split].tolist()
        if max_samples is not None:
            indices = indices[:max_samples]
        subset = IndexedGraphDataset(dataset, indices)
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def load_feature_tensor(path: Path, num_graphs: int, dim: int) -> torch.Tensor:
    """Load cached feature tensor or return zeros if missing."""
    if path.exists():
        obj = torch.load(path, weights_only=False)
        if isinstance(obj, dict) and "features" in obj:
            return obj["features"].float()
        return obj.float()
    return torch.zeros((num_graphs, dim), dtype=torch.float32)


def load_scalar_stats(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def normalize_train_stats(
    values: torch.Tensor, train_idx: torch.Tensor
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Z-score normalize using training set statistics only."""
    train_vals = values[train_idx].float()
    if train_vals.ndim == 1:
        mean = train_vals.mean()
        std = train_vals.std(unbiased=False).clamp_min(1e-8)
        normalized = (values - mean) / std
    else:
        mean = train_vals.mean(dim=0)
        std = train_vals.std(dim=0, unbiased=False).clamp_min(1e-8)
        normalized = (values - mean) / std
    stats = {"mean": mean, "std": std}
    return normalized, stats
