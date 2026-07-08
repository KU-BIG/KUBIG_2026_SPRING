# MolHIV TDA Research Prototype

Research prototype for applying TDA-based filtration features to molecular graph learning on **OGBG-MolHIV**.

**Backbone: PDGNN** (Persistence Diagram GNN from [TLC-GNN](https://github.com/pkuyzy/TLC-GNN)), adapted for OGB molecular graph classification.

## Project layout

```text
molhiv_tda/
├── models/
│   ├── pdgnn_conv.py       # PDConv layer (sum+min aggregation)
│   ├── pdgnn_baseline.py   # PDGNN backbone
│   ├── pdgnn_tda.py        # PDGNN + TDA fusion
│   └── han_molecule.py     # Pipeline 2 (HAN)
├── train/
│   ├── train_pdgnn.py
│   ├── train_pdgnn_tda.py
│   └── train_han.py
└── ...
```

## Setup

```bash
cd /home/jhkim/tda-conference
source .venv/bin/activate
cd molhiv_tda
```

## GPU setup

Your environment has **CUDA 12.6** (WSL2). Install CUDA PyTorch once:

```bash
bash scripts/setup_gpu.sh
```

Then train (defaults to GPU when available):

```bash
python train/train_pdgnn.py --epochs 100
python train/train_pdgnn_tda.py --config pdgnn_bond_tda_mw
python scripts/run_ablation.py --epochs 100 --device cuda
```


```bash
python scripts/preprocess_molecular_weight.py
python scripts/preprocess_bond_tda.py
# Baseline PDGNN (auto-detects GPU)
python train/train_pdgnn.py --epochs 100

# Force GPU
python train/train_pdgnn.py --device cuda --epochs 100
```

Quick dev:

```bash
python train/train_pdgnn.py --epochs 2 --max-samples 256
```

## PDGNN backbone

From TLC-GNN `Knowledge_Distillation/PD_conv.py`:

- `AtomEncoder` / `BondEncoder` for OGB features
- 4× `PDConv` with Union-Find style `concat(sum, min)` aggregation
- Messages: `lin_ij([x_i, x_j, bond_emb])`
- Global add pooling → MLP classifier

## Ablation table

| Model | Bond-type TDA | MW | 3D TDA | Valid AUC | Test AUC |
| --- | --- | --- | --- | ---: | ---: |
| PDGNN | No | No | No | | |
| PDGNN + MW | No | Yes | No | | |
| PDGNN + BondTDA | Yes | No | No | | |
| PDGNN + BondTDA + MW | Yes | Yes | No | | |
| PDGNN + 3DTDA | No | No | Yes | | |
| PDGNN + BondTDA + 3DTDA + MW | Yes | Yes | Yes | | |

## Available models

### Pipeline 1 — PDGNN (primary)

Homogeneous molecular graph + optional precomputed TDA features concatenated before the classifier head.

| Script | Config / entry | Features |
| --- | --- | --- |
| `train/train_pdgnn.py` | *(none)* | PDGNN backbone only |
| `train/train_pdgnn_tda.py` | `pdgnn_mw` | + molecular weight |
| `train/train_pdgnn_tda.py` | `pdgnn_bond_tda` | + bond-type filtration TDA |
| `train/train_pdgnn_tda.py` | `pdgnn_bond_tda_mw` | + bond TDA + MW |
| `train/train_pdgnn_tda.py` | `pdgnn_3d_tda` | + 3D conformer Rips TDA |
| `train/train_pdgnn_tda.py` | `pdgnn_bond_tda_3d_mw` | all three feature types |

Run all PDGNN ablations: `python scripts/run_ablation.py --epochs 100 --device cuda`

### Pipeline 2 — HAN (bond-relation heterograph)

Treats each bond type (single, double, triple, aromatic, misc) as a separate relation in a heterogeneous graph. Optional bond TDA and/or MW are concatenated at the head (no 3D TDA).

| Script | Config | Features |
| --- | --- | --- |
| `train/train_han.py` | `han` | HAN only |
| `train/train_han.py` | `han_mw` | + molecular weight |
| `train/train_han.py` | `han_bond_tda` | + bond-type filtration TDA |
| `train/train_han.py` | `han_bond_tda_mw` | + bond TDA + MW |

### Feature preprocessing

Required caches depend on the model config:

```bash
python scripts/preprocess_molecular_weight.py   # MW
python scripts/preprocess_bond_tda.py           # bond-type TDA
python scripts/preprocess_3d_tda.py             # 3D TDA (PDGNN only)
```
