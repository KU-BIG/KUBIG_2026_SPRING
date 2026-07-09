"""Train an RNN (GRU) detector: given the ego's first-100-step visual_obs
sequence, predict whether an adversary is on the team (1) or not (0).

Independent of the overcooked env (only numpy + torch). Saves detector.pt with
state_dict, config, and input normalization stats (used at eval time).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.environ.get("MULTIHRI_ROOT", "/workspace/rl_project/multiHRI")
DATA = f"{ROOT}/adaptive/data/detector_dataset.npz"
OUT = f"{ROOT}/adaptive/detector.pt"


class Detector(nn.Module):
    def __init__(self, in_dim=1323, emb=128, hidden=128, layers=1):
        super().__init__()
        self.enc = nn.Linear(in_dim, emb)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):  # x: (B, T, in_dim)
        h = torch.relu(self.enc(x))
        out, _ = self.gru(h)
        return self.head(out[:, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(cli.data)
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
    N, T, D = X.shape
    print(f"data X={X.shape} y={y.shape} pos={int(y.sum())} dev={dev}", flush=True)

    # stratified split
    rng = np.random.default_rng(cli.seed)
    idx = np.arange(N)
    val_idx = []
    for c in (0, 1):
        ci = idx[y == c]
        rng.shuffle(ci)
        k = int(len(ci) * cli.val_frac)
        val_idx.extend(ci[:k].tolist())
    val_idx = np.array(sorted(val_idx))
    tr_mask = np.ones(N, bool)
    tr_mask[val_idx] = False
    Xtr, ytr, Xva, yva = X[tr_mask], y[tr_mask], X[val_idx], y[val_idx]

    # normalize with train stats (over N,T)
    mean = Xtr.reshape(-1, D).mean(0)
    std = Xtr.reshape(-1, D).std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev)
    yva_t = torch.tensor(yva, device=dev)
    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=cli.batch, shuffle=True)

    model = Detector(in_dim=D, hidden=cli.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cli.lr)
    lossf = nn.CrossEntropyLoss()

    best_va = 0.0
    for ep in range(cli.epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xtr_t).argmax(1) == ytr_t).float().mean().item()
            va_acc = (model(Xva_t).argmax(1) == yva_t).float().mean().item()
        best_va = max(best_va, va_acc)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"ep {ep + 1:3d} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}", flush=True)

    torch.save({
        "state_dict": model.state_dict(),
        "config": {"in_dim": D, "hidden": cli.hidden, "emb": 128, "layers": 1, "T": T},
        "mean": mean, "std": std,
        "val_acc": best_va,
    }, cli.out)
    print(f"SAVED {cli.out} best_val_acc={best_va:.4f}", flush=True)


if __name__ == "__main__":
    main()
