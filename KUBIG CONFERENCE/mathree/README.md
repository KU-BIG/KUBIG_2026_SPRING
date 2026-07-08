# mathree — PDGNN + TDA + HAN (MolHIV)

KUBIG Conference 2026 Spring. Topological Data Analysis + PDGNN + heterogeneous HAN
node-level fusion for OGBG-MolHIV HIV inhibition prediction.

## Repository layout

- `molhiv_tda/` — main experiment code (PDGNN+TDA backbone, HAN fine-tune, sweeps, eval)
- `pdgnn/` — PDGNN reference notebooks and hetero-GNN notes
- `requirements.txt`, `setup.sh` — environment setup

## Key models

| Model | Description |
|---|---|
| `pdgnn_tda_3d_elec` | PDGNN + BondTDA + 3D TDA + electrostatic edge |
| `pdgnn_han_finetune_3d_elec_multifilt` | Frozen backbone + typed HAN + multifiltration (A/B) |

## Results summary (latest)

See `molhiv_tda/docs/pdgnn_han_finetune_multifilt_comparison.md`.

| Model | VALID | TEST full |
|---|---:|---:|
| Backbone `pdgnn_tda_3d_elec` | 0.7858 | **0.7761** |
| Fine-tune (multifilt + typed HAN) | **0.8488** | 0.7756 |

## Quick start

```bash
bash setup.sh
cd molhiv_tda
# preprocess features, then train — see molhiv_tda/README.md
```

## Note

Large artifacts (dataset, cache, `.pt` checkpoints) are excluded; regenerate via preprocess scripts on GPU.

Origin: elice server `~/jonghyun/Topological-Data-Analysis` (branch `feature/han-pdgnn-fusion`).
