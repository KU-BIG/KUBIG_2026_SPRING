#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDGNN="$ROOT/pdgnn"
TLC_GNN="$PDGNN/TLC-GNN"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip wheel setuptools

# PyTorch (CPU by default; for CUDA replace cpu with cu126/cu130/cu132)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# PyG 2.8+ (pyg-lib replaces torch-cluster/torch-spline-conv)
pip install pyg-lib torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
pip install torch-geometric

pip install gudhi networkx scikit-learn scipy numpy GraphRicciCurvature cython

# pdgnn/ has only gudhi patch files; clone upstream full code under pdgnn/
if [[ ! -d "$TLC_GNN" ]]; then
  git clone https://github.com/pkuyzy/TLC-GNN.git "$TLC_GNN"
fi

# Apply gudhi patches: dionysus -> gudhi (Python 3.11+ compatible)
cp "$PDGNN/dgformat.py" "$TLC_GNN/sg2dgm/dgformat.py"
cp "$PDGNN/riccidist2dgm.py" "$TLC_GNN/sg2dgm/riccidist2dgm.py"

# Build PersistenceImager Cython extension
cd "$TLC_GNN/sg2dgm"
python setup_PI.py build_ext --inplace || {
  echo "Cython build failed; falling back to .pyx copy"
  cp PersistenceImager.pyx PersistenceImager.py
}
cd "$ROOT"

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Run PDGNN pipeline: cd pdgnn/TLC-GNN && python pipelines.py"
