# HAN + PDGNN Cross-Attention Fusion (OGBG-MolHIV)

Research prototype combining:

1. **HAN** over exact atomic-number heterogeneous graphs
2. **PDGNN-style** bond-structure filtration encoder
3. **Node fusion** `l_i = concat(z_i, h_i^K)`
4. **Graph-wise cross-attention** over fused node latents
5. **Graph-level classification** (ROC-AUC)

## Architecture

```text
Molecular graph
    ↓
Exact heterogeneous node typing (atom_Z_6, atom_Z_7, ...)
    ↓
HAN encoder → z_i
    ↓
PDGNN bond-filtration encoder → h_i^K
    ↓
l_i = FusionMLP(concat(z_i, h_i^K))
    ↓
Cross-attention (Q from L, K/V from Z and H streams)
    ↓
Graph pooling → MLP classifier
```

## Why exact atomic number?

Coarse bins (heavy/light) lose chemical identity. OGB atom features encode atomic number as a categorical column; we decode it to `atom_Z_<Z>` node types so each element keeps its own heterogeneous node type.

## Bond filtration (PDGNN)

Bond types map to filtration scores (default):

| Bond | Score |
|------|------:|
| single | 1.0 |
| aromatic | 1.5 |
| double | 2.0 |
| triple | 3.0 |

Thresholds `tau ∈ {1.0, 1.5, 2.0, 3.0}` define subgraphs `G_tau = {e : f(e) ≤ tau}`. Per-node statistics at each threshold initialize PDGNN features; message passing uses SUM + MIN aggregation with bond-aware messages.

## Setup

```bash
cd /home/jhkim/tda-conference
source .venv/bin/activate
cd molhiv_tda/han_pdgnn_fusion
pip install pyyaml  # if not installed
```

**Cloud (vast.ai):** see [VASTAI.md](VASTAI.md) — `bash setup_vastai.sh` installs CUDA PyTorch, PyG, and downloads MolHIV automatically.

## Training

Quick smoke test:

```bash
python train_han_pdgnn.py --model main --max-samples 128 --config config.yaml
```

Train main model:

```bash
python train_han_pdgnn.py --model main --epochs 100
```

Run all baselines:

```bash
python main_han_pdgnn.py --model all --epochs 100
```

## Models

| `--model` | Description |
|-----------|-------------|
| `gcn` | Simple GCN baseline |
| `han_only` | HAN encoder only |
| `pdgnn_only` | PDGNN filtration encoder only |
| `concat` | HAN + PDGNN concat, no cross-attention |
| `main` | HAN + PDGNN + cross-attention (full model) |

## Config (`config.yaml`)

Key hyperparameters: `hidden_dim`, `model_dim`, `num_han_layers`, `num_pdg_layers`, `filtration_taus`, `node_type_mode` (`atomic_number` or `atomic_mass`), `attention_mode` (`cross` or `self`).

## Project layout

```text
han_pdgnn_fusion/
  main_han_pdgnn.py
  train_han_pdgnn.py
  dataset.py
  hetero_transform.py
  filtration.py
  models/
    han_encoder.py
    pdgnn_filtration_encoder.py
    cross_attention_fusion.py
    han_pdgnn_cross_attention.py
    baselines.py
  utils.py
  config.yaml
```

## Modular extension points

- `hetero_transform.py`: node/edge typing, optional MW supernode
- `filtration.py`: bond score mapping, threshold features, PDGNN layers
- `cross_attention_fusion.py`: fusion + attention mechanism
- Pooling mode in `config.yaml` (`mean` / `add` / `max`)
