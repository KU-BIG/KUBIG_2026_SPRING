#!/usr/bin/env bash
# Install CUDA PyTorch + PyG for GPU training (vast.ai / WSL CUDA 12.x).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/../.venv"

TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
CUDA_TAG="${CUDA_TAG:-cu124}"

if [[ ! -d "$VENV" ]]; then
  echo "Create venv first: python3 -m venv $VENV"
  exit 1
fi
source "$VENV/bin/activate"

pip install --upgrade pip wheel setuptools

echo "Removing old PyTorch / PyG wheels (if any)..."
pip uninstall -y pyg-lib torch-scatter torch-sparse torch-geometric torch torchvision 2>/dev/null || true

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "Installing PyTorch ${TORCH_VERSION} (${CUDA_TAG})..."
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX"

echo "Installing PyG extensions (${CUDA_TAG})..."
pip install pyg-lib torch-scatter torch-sparse -f "$PYG_INDEX"
pip install torch-geometric

python - <<'PY'
import torch
import torch_scatter
import torch_sparse
from torch_geometric.utils import scatter

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
scatter(torch.tensor([1.0, 2.0]), torch.tensor([0, 1]), dim=0, dim_size=2)
print("PyG OK")
PY

echo "Done. Run training with: python train/train_pdgnn.py --device cuda"
