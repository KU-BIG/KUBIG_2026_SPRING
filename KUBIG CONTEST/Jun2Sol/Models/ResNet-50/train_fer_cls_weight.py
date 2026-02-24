#!/usr/bin/env python3
"""
FER with Class-Weighted Loss to Improve Minority Class Performance
==================================================================

Motivation
----------
The GCN_FiLM model achieves 87.12% overall accuracy but struggles with
minority classes due to severe class imbalance:
  - disgust : ~1,300 training samples  → 54.2% accuracy
  - fear    : ~980  training samples  → 58.5% accuracy
  - happy   : ~9,600 training samples → 95.4% accuracy

This script trains GCN_FiLM_L34 with inverse-frequency class weighting,
which improves minority class accuracy at a small cost to overall accuracy:
  - disgust : 54.2% → 61.9%  (+7.7%)
  - fear    : 58.5% → 60.2%  (+1.7%)
  - overall : 87.12% → 86.37% (-0.8%)

Usage
-----
  python train_fer_cls_weight.py --data-root ./cleaned_7class --gpus 0,1

Requirements
------------
  pip install torch torchvision mediapipe scipy tqdm
  VGGFace2 pretrained weights: pretrained/resnet50_ft_weight.pkl
"""

import argparse
import copy
import json
import os
import pickle
import random
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

# ══════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════

N_LANDMARKS   = 478
LM_DIM        = N_LANDMARKS * 2
GNN_OUT_DIM   = 256
NUM_CLASSES   = 7
CLASS_NAMES   = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
VGGFACE2_PKL  = os.path.join(os.path.dirname(__file__), "pretrained", "resnet50_ft_weight.pkl")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════
#  Class-Weighted Loss
# ══════════════════════════════════════════════════════════════

def compute_class_weights(train_samples: list) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training samples.

    Weight for class c = (1 / count_c) normalized so that mean weight = 1.
    This up-weights minority classes (disgust, fear) and down-weights
    majority classes (happy, neutral).

    Returns:
        weights tensor of shape (NUM_CLASSES,)
    """
    counts = torch.zeros(NUM_CLASSES)
    for _, label in train_samples:
        counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * NUM_CLASSES  # normalize
    print("  Class weights (inverse frequency):")
    for i, (cls, w, cnt) in enumerate(zip(CLASS_NAMES, weights, counts)):
        print(f"    {cls:<10}  n={int(cnt):>5}  weight={w:.4f}")
    return weights


# ══════════════════════════════════════════════════════════════
#  GNN Modules  (identical to train_fer_gcn_film.py)
# ══════════════════════════════════════════════════════════════

def build_knn_adj(landmarks_mean: np.ndarray, k: int = 8) -> torch.Tensor:
    from scipy.spatial import cKDTree
    tree = cKDTree(landmarks_mean)
    _, idx = tree.query(landmarks_mean, k=k + 1)
    idx = idx[:, 1:]
    adj = np.zeros((N_LANDMARKS, N_LANDMARKS), dtype=np.float32)
    for i in range(N_LANDMARKS):
        adj[i, idx[i]] = 1.0
    adj = np.maximum(adj, adj.T)
    adj += np.eye(N_LANDMARKS)
    deg = adj.sum(1)
    d_inv = np.where(deg > 0, deg ** -0.5, 0.0)
    return torch.from_numpy(d_inv[:, None] * adj * d_inv[None, :])


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        return torch.matmul(adj, self.linear(x))


class LandmarkGCN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, out_dim=GNN_OUT_DIM,
                 n_layers=3, adj=None):
        super().__init__()
        self.register_buffer("adj", adj if adj is not None else torch.eye(N_LANDMARKS))
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers - 1)])

    def forward(self, lm_flat):
        x = lm_flat.view(-1, N_LANDMARKS, 2)
        for i, layer in enumerate(self.layers):
            x = layer(x, self.adj)
            if i < len(self.layers) - 1:
                x = F.relu(self.norms[i](x))
                x = F.dropout(x, p=0.1, training=self.training)
        return x.mean(dim=1)


# ══════════════════════════════════════════════════════════════
#  FiLM Modules  (identical to train_fer_gcn_film.py)
# ══════════════════════════════════════════════════════════════

class FiLM2d(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, num_features))
        self.beta_fc = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, num_features))

    def forward(self, x, cond):
        g = self.gamma_fc(cond).view(x.size(0), -1, 1, 1)
        b = self.beta_fc(cond).view(x.size(0), -1, 1, 1)
        return g * x + b


class FiLMBottleneck(nn.Module):
    def __init__(self, original, cond_dim):
        super().__init__()
        self.conv1 = original.conv1; self.bn1 = original.bn1
        self.film1 = FiLM2d(original.bn1.num_features, cond_dim)
        self.conv2 = original.conv2; self.bn2 = original.bn2
        self.film2 = FiLM2d(original.bn2.num_features, cond_dim)
        self.conv3 = original.conv3; self.bn3 = original.bn3
        self.film3 = FiLM2d(original.bn3.num_features, cond_dim)
        self.relu   = original.relu
        self.stride = original.stride
        self.downsample_conv = self.downsample_bn = self.downsample_film = None
        if original.downsample is not None:
            self.downsample_conv = original.downsample[0]
            self.downsample_bn   = original.downsample[1]
            self.downsample_film = FiLM2d(original.downsample[1].num_features, cond_dim)

    def forward(self, x, cond):
        identity = x
        out = self.relu(self.film1(self.bn1(self.conv1(x)), cond))
        out = self.relu(self.film2(self.bn2(self.conv2(out)), cond))
        out = self.film3(self.bn3(self.conv3(out)), cond)
        if self.downsample_conv is not None:
            identity = self.downsample_film(
                self.downsample_bn(self.downsample_conv(x)), cond)
        return self.relu(out + identity)


class FiLMLayer(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)
        return x


# ══════════════════════════════════════════════════════════════
#  Model: GCN_FiLM_L34
# ══════════════════════════════════════════════════════════════

def _load_resnet50_vggface2() -> nn.Module:
    backbone = models.resnet50(weights=None)
    if os.path.exists(VGGFACE2_PKL):
        with open(VGGFACE2_PKL, "rb") as f:
            weights = pickle.load(f, encoding="latin1")
        sd = backbone.state_dict()
        n = sum(1 for k, v in weights.items()
                if k in sd and sd[k].shape == torch.from_numpy(v).shape
                and not sd.__setitem__(k, torch.from_numpy(v)))
        backbone.load_state_dict(sd)
        print(f"  VGGFace2 weights loaded: {n}/{len(weights)}")
    else:
        print(f"  [WARN] VGGFace2 weights not found. Using ImageNet init.")
    return backbone


class GCNFiLM_L34(nn.Module):
    """
    GCN_FiLM_L34: Best-performing model from the ablation study.

    - GCN landmark encoder (3-layer, k=8 k-NN graph, 256-dim output)
    - FiLM conditioning applied at layer3 and layer4 of ResNet-50
    - Test accuracy: 87.12% (vs Baseline 86.18%)
    """
    def __init__(self, num_classes: int = NUM_CLASSES, adj: torch.Tensor = None):
        super().__init__()
        self.gnn = LandmarkGCN(adj=adj)
        bb = _load_resnet50_vggface2()
        self.stem    = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1  = bb.layer1
        self.layer2  = bb.layer2
        self.layer3  = FiLMLayer([FiLMBottleneck(b, GNN_OUT_DIM) for b in bb.layer3.children()])
        self.layer4  = FiLMLayer([FiLMBottleneck(b, GNN_OUT_DIM) for b in bb.layer4.children()])
        self.avgpool = bb.avgpool
        self.fc      = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
        cond = self.gnn(lm)
        x    = self.stem(x)
        x    = self.layer1(x)
        x    = self.layer2(x)
        x    = self.layer3(x, cond)
        x    = self.layer4(x, cond)
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ══════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════

def extract_landmarks(data_root: str) -> np.ndarray:
    cache_path = os.path.join(data_root, "landmarks.npy")
    if os.path.exists(cache_path):
        lm = np.load(cache_path)
        print(f"  Landmarks loaded: {cache_path}  shape={lm.shape}")
        return lm

    import mediapipe as mp_lib
    model_path = os.path.join(data_root, "face_landmarker_v2_with_blendshapes.task")
    if not os.path.exists(model_path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        urllib.request.urlretrieve(url, model_path)

    BaseOptions           = mp_lib.tasks.BaseOptions
    FaceLandmarker        = mp_lib.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_lib.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp_lib.tasks.vision.RunningMode

    ds = ImageFolder(data_root)
    all_lm, n_failed = [], 0
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.2)
    with FaceLandmarker.create_from_options(opts) as lm_model:
        for img_path, _ in tqdm(ds.samples, desc="Extracting landmarks"):
            try:
                img    = Image.open(img_path).convert("RGB")
                mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB,
                                      data=np.array(img))
                result = lm_model.detect(mp_img)
                if result.face_landmarks:
                    coords = np.array([[lm.x, lm.y]
                                       for lm in result.face_landmarks[0]], dtype=np.float32)
                    all_lm.append(coords.flatten())
                else:
                    all_lm.append(np.zeros(LM_DIM, dtype=np.float32)); n_failed += 1
            except Exception:
                all_lm.append(np.zeros(LM_DIM, dtype=np.float32)); n_failed += 1
    landmarks = np.stack(all_lm)
    np.save(cache_path, landmarks)
    print(f"  Extracted {len(ds)} landmarks. Failed: {n_failed}")
    return landmarks


class FERLandmarkDataset(Dataset):
    def __init__(self, samples, landmarks, transform):
        self.samples = samples; self.landmarks = landmarks; self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.from_numpy(self.landmarks[idx]).float(), label


def get_dataloaders(data_root, all_landmarks, batch_size=128, num_workers=4):
    full_ds     = ImageFolder(data_root)
    all_samples = full_ds.samples
    n_total     = len(all_samples)
    n_test = int(n_total * 0.1); n_val = int(n_total * 0.1)
    n_train = n_total - n_val - n_test

    gen     = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    train_samples = [all_samples[i] for i in train_idx]
    train_ds = FERLandmarkDataset(train_samples, all_landmarks[train_idx], train_tfm)
    val_ds   = FERLandmarkDataset([all_samples[i] for i in val_idx],
                                   all_landmarks[val_idx], val_tfm)
    test_ds  = FERLandmarkDataset([all_samples[i] for i in test_idx],
                                   all_landmarks[test_idx], val_tfm)

    print(f"  Split: Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    kw = dict(num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
            train_samples)


# ══════════════════════════════════════════════════════════════
#  Training & Evaluation
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, lm, labels in tqdm(loader, desc="  Train", leave=False):
        images, lm, labels = (images.to(device, non_blocking=True),
                               lm.to(device, non_blocking=True),
                               labels.to(device, non_blocking=True))
        optimizer.zero_grad()
        logits = model(images, lm)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, lm, labels in loader:
        images, lm, labels = (images.to(device, non_blocking=True),
                               lm.to(device, non_blocking=True),
                               labels.to(device, non_blocking=True))
        logits    = model(images, lm)
        loss_sum += criterion(logits, labels).item() * images.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_per_class(model, loader, device):
    model.eval()
    correct_per = torch.zeros(NUM_CLASSES)
    total_per   = torch.zeros(NUM_CLASSES)
    for images, lm, labels in loader:
        images, lm, labels = (images.to(device, non_blocking=True),
                               lm.to(device, non_blocking=True),
                               labels.to(device, non_blocking=True))
        preds = model(images, lm).argmax(1)
        for c in range(NUM_CLASSES):
            mask = (labels == c)
            total_per[c]   += mask.sum()
            correct_per[c] += (preds[mask] == c).sum()
    overall   = correct_per.sum().item() / total_per.sum().item()
    per_class = {
        CLASS_NAMES[c]: {"accuracy": (correct_per[c] / total_per[c]).item(),
                          "correct":  int(correct_per[c]),
                          "total":    int(total_per[c])}
        for c in range(NUM_CLASSES) if total_per[c] > 0}
    return overall, per_class


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GCN_FiLM_L34 training with class-weighted loss for minority class improvement")
    parser.add_argument("--data-root",    type=str,   default="./cleaned_7class")
    parser.add_argument("--save-dir",     type=str,   default="./results_cls_weight")
    parser.add_argument("--gpus",         type=str,   default="0",
                        help="Comma-separated GPU IDs (e.g. 0,1)")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--batch-size",   type=int,   default=128)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--num-workers",  type=int,   default=4)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    device  = torch.device(f"cuda:{gpu_ids[0]}")
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    print(f"[GCN_FiLM_L34 + Class-Weighted Loss]")
    print(f"GPUs: {gpu_ids}  Epochs: {args.epochs}  LR: {args.lr}")

    # Landmarks
    all_lm = extract_landmarks(args.data_root)

    # Data loaders
    train_loader, val_loader, test_loader, train_samples = get_dataloaders(
        args.data_root, all_lm, args.batch_size, args.num_workers)

    # Class-weighted loss
    class_weights = compute_class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smooth)

    # Adjacency matrix from training data mean positions
    full_ds = ImageFolder(args.data_root)
    gen     = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_ds.samples), generator=gen).tolist()
    n_train = len(full_ds.samples) - 2 * int(len(full_ds.samples) * 0.1)
    train_lm_mean = all_lm[indices[:n_train]].reshape(-1, N_LANDMARKS, 2).mean(0)
    adj = build_knn_adj(train_lm_mean, k=8).to(device)

    # Model
    model = GCNFiLM_L34(adj=adj).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"  DataParallel on GPUs {gpu_ids}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val, best_state, best_epoch = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        for k, v in zip(history, [tr_loss, vl_loss, tr_acc, vl_acc]):
            history[k].append(v)

        mark = ""
        if vl_acc > best_val:
            best_val   = vl_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(
                model.module.state_dict() if hasattr(model, "module") else model.state_dict())
            mark = " ★"

        print(f"  Epoch [{epoch+1:>2d}/{args.epochs}]  "
              f"Train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"Val {vl_loss:.4f}/{vl_acc:.4f}{mark}")

    # Save checkpoint
    ckpt_path = os.path.join(args.save_dir, "best_GCN_FiLM_L34_cls_weight.pt")
    torch.save(best_state, ckpt_path)
    print(f"\n  Saved → {ckpt_path}")

    # Evaluate
    eval_model = GCNFiLM_L34(adj=adj).to(device)
    eval_model.load_state_dict(best_state)
    test_acc, per_class = evaluate_per_class(eval_model, test_loader, device)
    elapsed = (time.time() - t0) / 60

    print(f"\n  Val: {best_val:.4f}  Test: {test_acc:.4f}  "
          f"Best epoch: {best_epoch}  ({elapsed:.1f} min)")
    print("\n  Per-class accuracy:")
    baseline = {"angry":0.834,"disgust":0.542,"fear":0.585,"happy":0.954,
                "neutral":0.879,"sad":0.866,"surprise":0.880}
    for cls in CLASS_NAMES:
        if cls in per_class:
            info = per_class[cls]
            diff = info["accuracy"] - baseline.get(cls, 0)
            sign = "+" if diff >= 0 else ""
            print(f"    {cls:<10}  {info['accuracy']:.4f}  "
                  f"({info['correct']}/{info['total']})  "
                  f"[vs baseline: {sign}{diff:+.3f}]")

    result = {"test_acc": test_acc, "best_val_acc": best_val, "per_class": per_class,
              "n_params": n_params, "best_epoch": best_epoch,
              "actual_epochs": args.epochs, "history": history, "time_min": elapsed}
    result_path = os.path.join(args.save_dir, "result_cls_weight.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {result_path}")


if __name__ == "__main__":
    main()
