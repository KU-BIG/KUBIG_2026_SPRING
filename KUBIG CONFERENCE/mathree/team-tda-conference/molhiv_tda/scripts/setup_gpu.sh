#!/usr/bin/env bash
# Install CUDA PyTorch + PyG for GPU training (CUDA 12.6).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/../.venv"

if [[ ! -d "$VENV" ]]; then
  echo "Create venv first: python3 -m venv $VENV"
  exit 1
fi
source "$VENV/bin/activate"

CUDA_TAG="${CUDA_TAG:-cu126}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
PYG_INDEX="https://data.pyg.org/whl/torch-2.12.0+${CUDA_TAG}.html"

echo "Installing PyTorch (${CUDA_TAG})..."
pip install --upgrade pip
pip install torch torchvision --index-url "$TORCH_INDEX"

echo "Installing PyG extensions (${CUDA_TAG})..."
pip install pyg-lib torch-scatter torch-sparse -f "$PYG_INDEX"
pip install torch-geometric

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Done. Run training with: python train/train_pdgnn.py --device cuda"
