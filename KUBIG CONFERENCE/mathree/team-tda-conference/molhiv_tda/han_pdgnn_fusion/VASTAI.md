# Running HAN+PDGNN on vast.ai

## 1. Push code to GitHub

From your machine:

```bash
cd ~/tda-conference
git add molhiv_tda/han_pdgnn_fusion
git commit -m "Add han_pdgnn_fusion vast.ai setup"
git push
```

You do **not** need to upload `dataset/` or `.venv/` — OGB downloads on the instance.

## 2. Rent a vast.ai instance

Recommended filters:

| Setting | Value |
|---------|--------|
| Template | `PyTorch` or `Ubuntu 22.04` |
| GPU | RTX 3090 / 4090 / A5000 (≥16 GB VRAM) |
| Disk | ≥30 GB |
| CUDA | 12.x |

## 3. On-start command (paste in vast.ai “On-start script”)

Replace `YOUR_USER/YOUR_REPO` with your GitHub repo:

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git /workspace/tda-conference
cd /workspace/tda-conference/molhiv_tda/han_pdgnn_fusion
bash setup_vastai.sh
```

## 4. Train (SSH into the instance)

```bash
source /workspace/tda-conference/.venv/bin/activate
cd /workspace/tda-conference/molhiv_tda/han_pdgnn_fusion

# Quick test (~5 min)
python train_han_pdgnn.py --model main --epochs 2 --max-samples 512 --device cuda

# Full main model (~1.5–3 h for 10 epochs, ~half day for 100)
python train_han_pdgnn.py --model main --epochs 100 --device cuda

# All 5 ablation models
python main_han_pdgnn.py --model all --epochs 100 --device cuda
```

## 5. Run training in background (recommended)

```bash
nohup python train_han_pdgnn.py --model main --epochs 100 --device cuda \
  > train_main.log 2>&1 &
tail -f train_main.log
```

## 6. Download results

From your local machine:

```bash
scp -P PORT root@IP:/workspace/tda-conference/molhiv_tda/han_pdgnn_fusion/results/* ./results/
```

Results:

- `results/han_pdgnn_main.json` — metrics
- `results/han_pdgnn_main_best.pt` — best checkpoint

## 7. One-liner on-start (clone + setup + train)

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git /workspace/tda-conference && \
cd /workspace/tda-conference/molhiv_tda/han_pdgnn_fusion && \
bash setup_vastai.sh && \
source /workspace/tda-conference/.venv/bin/activate && \
nohup python train_han_pdgnn.py --model main --epochs 100 --device cuda > train.log 2>&1 &
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Dataset not found` | `python download_molhiv.py` |
| CUDA mismatch | `CUDA_TAG=cu124 bash setup_vastai.sh` |
| OOM | `--batch-size 16` or smaller model in `config.yaml` |
| Slow first epoch | Normal (HAN runs per-molecule); wait 5–15 min for `Epoch 001` |

## Cost estimate

| GPU | ~10 epochs (main) | ~100 epochs |
|-----|-------------------|-------------|
| RTX 3090 (~$0.20/hr) | $0.50–1.50 | $5–15 |
| RTX 4090 (~$0.35/hr) | $0.40–1.00 | $4–10 |
