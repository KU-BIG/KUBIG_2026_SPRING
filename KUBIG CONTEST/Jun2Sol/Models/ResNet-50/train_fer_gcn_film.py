#!/usr/bin/env python3
"""
Facial Expression Recognition with GCN Landmark Encoder + FiLM Conditioning
============================================================================

Architecture
------------
- Backbone  : ResNet-50 pretrained on VGGFace2 (face-specialized)
- Landmark Encoder : GCN / GAT over 478 MediaPipe face landmarks → 256-dim embedding
- Conditioning     : FiLM (Feature-wise Linear Modulation) injected at ResNet layers

Best result: GCN_FiLM_L34 — Test Acc 87.12%
  - GCN encoder (3-layer, k=8 NN graph)
  - FiLM conditioning at layer3 + layer4 of ResNet-50

Ablation study (8 models):
  GCN/GAT  ×  FiLM  ×  L4 / L34 / L234 / All

Usage
-----
  # Run full ablation on 8 GPUs (1 model per GPU)
  python train_fer_gcn_film.py --data-root ./cleaned_7class --gpus 0,1,2,3,4,5,6,7

  # Run specific experiments on 2 GPUs
  python train_fer_gcn_film.py --data-root ./cleaned_7class --gpus 0,1 \\
      --experiments GCN_FiLM_L34 GCN_FiLM_L4

  # Single GPU
  python train_fer_gcn_film.py --data-root ./cleaned_7class --gpus 0 \\
      --experiments GCN_FiLM_L34

Requirements
------------
  pip install torch torchvision mediapipe scipy tqdm
  VGGFace2 pretrained weights: pretrained/resnet50_ft_weight.pkl
    → Download: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
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
LM_DIM        = N_LANDMARKS * 2   # flattened (x, y) coordinates
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
#  Graph Construction
# ══════════════════════════════════════════════════════════════

def build_knn_adj(landmarks_mean: np.ndarray, k: int = 8) -> torch.Tensor:
    """
    Build a normalized k-NN adjacency matrix from mean landmark positions.

    Args:
        landmarks_mean: (N_LANDMARKS, 2) mean (x, y) positions
        k: number of nearest neighbors

    Returns:
        Symmetric, degree-normalized adjacency matrix (N, N)
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(landmarks_mean)
    _, idx = tree.query(landmarks_mean, k=k + 1)
    idx = idx[:, 1:]  # exclude self

    adj = np.zeros((N_LANDMARKS, N_LANDMARKS), dtype=np.float32)
    for i in range(N_LANDMARKS):
        adj[i, idx[i]] = 1.0
    adj = np.maximum(adj, adj.T)           # symmetrize
    adj += np.eye(N_LANDMARKS)             # self-loops
    deg = adj.sum(1)
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]  # D^{-1/2} A D^{-1/2}
    return torch.from_numpy(adj)


# ══════════════════════════════════════════════════════════════
#  GNN Modules
# ══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer: X' = A_norm @ X @ W"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F), adj: (N, N)
        return torch.matmul(adj, self.linear(x))


class GATLayer(nn.Module):
    """Multi-head Graph Attention Layer."""
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_features % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = out_features // n_heads
        self.W        = nn.Linear(in_features, out_features, bias=False)
        self.a_src    = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.a_dst    = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.n_heads, self.head_dim)
        attn = self.leaky_relu(
            h.unsqueeze(2) * self.a_src + h.unsqueeze(1) * self.a_dst
        ).sum(-1)  # (B, N, N, H)
        attn = attn.masked_fill((adj == 0).unsqueeze(0).unsqueeze(-1), float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=2))
        out  = torch.matmul(attn.permute(0, 3, 1, 2),
                            h.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return out.reshape(B, N, -1)


class LandmarkGCN(nn.Module):
    """
    GCN-based landmark encoder.
    Input : (B, 956)  — flattened 478 × (x, y)
    Output: (B, 256)  — global mean-pooled graph embedding
    """
    def __init__(self, in_dim: int = 2, hidden_dim: int = 64,
                 out_dim: int = GNN_OUT_DIM, n_layers: int = 3,
                 k: int = 8, adj: torch.Tensor = None):
        super().__init__()
        self.register_buffer("adj", adj if adj is not None else torch.eye(N_LANDMARKS))
        dims   = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers - 1)]
        )

    def forward(self, lm_flat: torch.Tensor) -> torch.Tensor:
        x = lm_flat.view(-1, N_LANDMARKS, 2)
        for i, layer in enumerate(self.layers):
            x = layer(x, self.adj)
            if i < len(self.layers) - 1:
                x = F.relu(self.norms[i](x))
                x = F.dropout(x, p=0.1, training=self.training)
        return x.mean(dim=1)  # global mean pool → (B, out_dim)


class LandmarkGAT(nn.Module):
    """
    GAT-based landmark encoder.
    Input : (B, 956)
    Output: (B, 256)
    """
    def __init__(self, in_dim: int = 2, hidden_dim: int = 64,
                 out_dim: int = GNN_OUT_DIM, n_layers: int = 3,
                 n_heads: int = 4, k: int = 8, adj: torch.Tensor = None):
        super().__init__()
        self.register_buffer("adj", adj if adj is not None else torch.eye(N_LANDMARKS))
        dims   = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GATLayer(dims[i], dims[i + 1], n_heads=n_heads) for i in range(n_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers - 1)]
        )

    def forward(self, lm_flat: torch.Tensor) -> torch.Tensor:
        x = lm_flat.view(-1, N_LANDMARKS, 2)
        for i, layer in enumerate(self.layers):
            x = layer(x, self.adj)
            if i < len(self.layers) - 1:
                x = F.elu(self.norms[i](x))
                x = F.dropout(x, p=0.1, training=self.training)
        return x.mean(dim=1)


# ══════════════════════════════════════════════════════════════
#  FiLM Conditioning Modules
# ══════════════════════════════════════════════════════════════

class FiLM2d(nn.Module):
    """
    Feature-wise Linear Modulation for 2D feature maps.
    Applies per-channel scale/shift conditioned on a vector:
        out = γ(c) * x + β(c)
    where γ, β are produced by small MLPs from condition vector c.
    """
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, num_features)
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, num_features)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.gamma_fc(cond).view(x.size(0), -1, 1, 1)
        b = self.beta_fc(cond).view(x.size(0), -1, 1, 1)
        return g * x + b


class FiLMBottleneck(nn.Module):
    """
    ResNet-50 Bottleneck with FiLM applied after each BatchNorm.
    Original BN is preserved; FiLM adds an additional scale/shift.
    """
    def __init__(self, original: nn.Module, cond_dim: int):
        super().__init__()
        self.conv1  = original.conv1
        self.bn1    = original.bn1
        self.film1  = FiLM2d(original.bn1.num_features, cond_dim)
        self.conv2  = original.conv2
        self.bn2    = original.bn2
        self.film2  = FiLM2d(original.bn2.num_features, cond_dim)
        self.conv3  = original.conv3
        self.bn3    = original.bn3
        self.film3  = FiLM2d(original.bn3.num_features, cond_dim)
        self.relu   = original.relu
        self.stride = original.stride
        self.downsample_conv = self.downsample_bn = self.downsample_film = None
        if original.downsample is not None:
            self.downsample_conv = original.downsample[0]
            self.downsample_bn   = original.downsample[1]
            self.downsample_film = FiLM2d(original.downsample[1].num_features, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.film1(self.bn1(self.conv1(x)), cond))
        out = self.relu(self.film2(self.bn2(self.conv2(out)), cond))
        out = self.film3(self.bn3(self.conv3(out)), cond)
        if self.downsample_conv is not None:
            identity = self.downsample_film(
                self.downsample_bn(self.downsample_conv(x)), cond)
        return self.relu(out + identity)


class FiLMLayer(nn.Module):
    """Sequential container for FiLMBottleneck blocks."""
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


# ══════════════════════════════════════════════════════════════
#  Backbone Loader
# ══════════════════════════════════════════════════════════════

def _load_resnet50_vggface2() -> nn.Module:
    """Load ResNet-50 with VGGFace2 pretrained weights (face-specialized)."""
    backbone = models.resnet50(weights=None)
    if os.path.exists(VGGFACE2_PKL):
        with open(VGGFACE2_PKL, "rb") as f:
            weights = pickle.load(f, encoding="latin1")
        sd = backbone.state_dict()
        n_loaded = sum(
            1 for k, v in weights.items()
            if k in sd and sd[k].shape == torch.from_numpy(v).shape
            and not sd.__setitem__(k, torch.from_numpy(v))
        )
        backbone.load_state_dict(sd)
        print(f"  VGGFace2 weights loaded: {n_loaded}/{len(weights)}")
    else:
        print(f"  [WARN] VGGFace2 weights not found at {VGGFACE2_PKL}. Using ImageNet init.")
    return backbone


# ══════════════════════════════════════════════════════════════
#  Models
# ══════════════════════════════════════════════════════════════

class BaselineFER(nn.Module):
    """
    Baseline: ResNet-50 (VGGFace2) + linear classifier.
    No landmark conditioning.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        bb = _load_resnet50_vggface2()
        bb.fc = nn.Identity()
        self.backbone = bb
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


class GNNFiLM_FER(nn.Module):
    """
    ResNet-50 (VGGFace2) conditioned by GNN landmark encoder via FiLM.

    Data flow:
        landmarks (B, 956) → GNN → embedding (B, 256)
        image     (B, 3, 224, 224) → ResNet stem → layer1 → ... → layer4
                                     ↑ FiLM at specified layers using embedding
        → avgpool → fc → logits (B, 7)

    Args:
        gnn_type   : "GCN" or "GAT"
        film_layers: subset of ["stem","layer1","layer2","layer3","layer4"]
        adj        : precomputed adjacency matrix (N, N)
    """
    def __init__(self, gnn_type: str, film_layers: list,
                 gnn_out_dim: int = GNN_OUT_DIM,
                 num_classes: int = NUM_CLASSES,
                 adj: torch.Tensor = None):
        super().__init__()

        # GNN encoder
        if gnn_type == "GCN":
            self.gnn = LandmarkGCN(out_dim=gnn_out_dim, adj=adj)
        elif gnn_type == "GAT":
            self.gnn = LandmarkGAT(out_dim=gnn_out_dim, adj=adj)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        bb = _load_resnet50_vggface2()

        # Stem
        if "stem" in film_layers:
            self.stem_conv  = nn.Sequential(bb.conv1)
            self.stem_bn    = bb.bn1
            self.stem_film  = FiLM2d(bb.bn1.num_features, gnn_out_dim)
            self.stem_relu  = bb.relu
            self.stem_pool  = bb.maxpool
            self._film_stem = True
        else:
            self.stem       = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
            self._film_stem = False

        # Residual layers
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            orig = getattr(bb, name)
            if name in film_layers:
                setattr(self, name, FiLMLayer([FiLMBottleneck(b, gnn_out_dim)
                                               for b in orig.children()]))
                setattr(self, f"_film_{name}", True)
            else:
                setattr(self, name, orig)
                setattr(self, f"_film_{name}", False)

        self.avgpool = bb.avgpool
        self.fc      = nn.Linear(2048, num_classes)

    def _forward_layer(self, layer, x, cond, flag):
        return layer(x, cond) if flag else layer(x)

    def forward(self, x: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
        cond = self.gnn(lm)

        if self._film_stem:
            x = self.stem_film(self.stem_bn(self.stem_conv(x)), cond)
            x = self.stem_pool(self.stem_relu(x))
        else:
            x = self.stem(x)

        x = self._forward_layer(self.layer1, x, cond, self._film_layer1)
        x = self._forward_layer(self.layer2, x, cond, self._film_layer2)
        x = self._forward_layer(self.layer3, x, cond, self._film_layer3)
        x = self._forward_layer(self.layer4, x, cond, self._film_layer4)

        return self.fc(torch.flatten(self.avgpool(x), 1))


# ══════════════════════════════════════════════════════════════
#  Dataset & Data Loading
# ══════════════════════════════════════════════════════════════

def extract_landmarks(data_root: str) -> np.ndarray:
    """
    Extract MediaPipe face landmarks for all images and cache to .npy.
    Loads from cache if already computed.

    Returns:
        landmarks array of shape (N_images, 956)
    """
    cache_path = os.path.join(data_root, "landmarks.npy")
    if os.path.exists(cache_path):
        lm = np.load(cache_path)
        print(f"  Landmarks loaded from cache: {cache_path}  shape={lm.shape}")
        return lm

    import mediapipe as mp_lib
    model_path = os.path.join(data_root, "face_landmarker_v2_with_blendshapes.task")
    if not os.path.exists(model_path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print(f"  Downloading MediaPipe model...")
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
                    all_lm.append(np.zeros(LM_DIM, dtype=np.float32))
                    n_failed += 1
            except Exception:
                all_lm.append(np.zeros(LM_DIM, dtype=np.float32))
                n_failed += 1

    landmarks = np.stack(all_lm)
    np.save(cache_path, landmarks)
    print(f"  Extracted {len(ds)} landmarks. Failed: {n_failed}")
    return landmarks


class FERLandmarkDataset(Dataset):
    def __init__(self, samples, landmarks, transform):
        self.samples   = samples
        self.landmarks = landmarks
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.from_numpy(self.landmarks[idx]).float(), label


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(data_root, all_landmarks, batch_size=128,
                    val_ratio=0.1, test_ratio=0.1, num_workers=4):
    full_ds     = ImageFolder(data_root)
    all_samples = full_ds.samples
    n_total     = len(all_samples)
    n_test      = int(n_total * test_ratio)
    n_val       = int(n_total * val_ratio)
    n_train     = n_total - n_val - n_test

    gen     = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    train_idx, val_idx, test_idx = (indices[:n_train],
                                    indices[n_train:n_train + n_val],
                                    indices[n_train + n_val:])

    train_ds = FERLandmarkDataset(
        [all_samples[i] for i in train_idx], all_landmarks[train_idx], get_transforms("train"))
    val_ds   = FERLandmarkDataset(
        [all_samples[i] for i in val_idx],   all_landmarks[val_idx],   get_transforms("test"))
    test_ds  = FERLandmarkDataset(
        [all_samples[i] for i in test_idx],  all_landmarks[test_idx],  get_transforms("test"))

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
            train_idx)


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
    per_class = {CLASS_NAMES[c]: {"accuracy": (correct_per[c] / total_per[c]).item(),
                                   "correct":  int(correct_per[c]),
                                   "total":    int(total_per[c])}
                 for c in range(NUM_CLASSES) if total_per[c] > 0}
    return overall, per_class


def run_experiment(name, model_fn, train_loader, val_loader, test_loader,
                   epochs, lr, label_smoothing, gpu_ids, save_dir):
    device = torch.device(f"cuda:{gpu_ids[0]}")
    print(f"\n{'='*60}\n  [{name}]  GPUs={gpu_ids}\n{'='*60}")

    set_seed(42)
    model    = model_fn().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, best_epoch = 0.0, None, 0
    t0 = time.time()

    for epoch in range(epochs):
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

        print(f"  Epoch [{epoch+1:>2d}/{epochs}]  "
              f"Train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"Val {vl_loss:.4f}/{vl_acc:.4f}  "
              f"LR={optimizer.param_groups[0]['lr']:.2e}{mark}")

    # Save best checkpoint
    ckpt_path = os.path.join(save_dir, f"best_{name}.pt")
    torch.save(best_state, ckpt_path)
    print(f"  Saved checkpoint → {ckpt_path}")

    # Evaluate on test set
    eval_model = model_fn().to(device)
    eval_model.load_state_dict(best_state)
    test_acc, per_class = evaluate_per_class(eval_model, test_loader, device)

    elapsed = (time.time() - t0) / 60
    print(f"  Val: {best_val:.4f}  Test: {test_acc:.4f}  "
          f"Best epoch: {best_epoch}  ({elapsed:.1f} min)")
    print("  Per-class accuracy:")
    for cls in CLASS_NAMES:
        if cls in per_class:
            info = per_class[cls]
            print(f"    {cls:<10}  {info['accuracy']:.4f}  ({info['correct']}/{info['total']})")

    return {"best_val_acc": best_val, "test_acc": test_acc, "per_class": per_class,
            "n_params": n_params, "best_epoch": best_epoch,
            "actual_epochs": epochs, "history": history, "time_min": elapsed}


# ══════════════════════════════════════════════════════════════
#  Experiment Factory
# ══════════════════════════════════════════════════════════════

LAYER_CONFIGS = {
    "L4":   ["layer4"],
    "L34":  ["layer3", "layer4"],
    "L234": ["layer2", "layer3", "layer4"],
    "All":  ["stem", "layer1", "layer2", "layer3", "layer4"],
}

ALL_EXPERIMENTS = (
    ["Baseline"]
    + [f"GCN_FiLM_{s}" for s in LAYER_CONFIGS]
    + [f"GAT_FiLM_{s}" for s in LAYER_CONFIGS]
)


def build_model_fn(name: str, adj: torch.Tensor):
    if name == "Baseline":
        return lambda: BaselineFER()
    gnn_type, _, suffix = name.split("_")
    layers = LAYER_CONFIGS[suffix]
    return lambda g=gnn_type, fl=layers: GNNFiLM_FER(g, fl, adj=adj)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FER GCN-FiLM Ablation Study")
    parser.add_argument("--data-root",    type=str, default="./cleaned_7class")
    parser.add_argument("--save-dir",     type=str, default="./results")
    parser.add_argument("--gpus",         type=str, default="0",
                        help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--experiments",  nargs="+", default=ALL_EXPERIMENTS,
                        help="Experiment names to run")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--batch-size",   type=int,   default=128)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--num-workers",  type=int,   default=4)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"GPUs: {gpu_ids}")
    print(f"Experiments: {args.experiments}")

    # Load/extract landmarks
    all_lm = extract_landmarks(args.data_root)

    # Build adjacency matrix from training data mean positions
    full_ds = ImageFolder(args.data_root)
    gen     = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_ds.samples), generator=gen).tolist()
    n_train = len(full_ds.samples) - 2 * int(len(full_ds.samples) * 0.1)
    train_lm_mean = all_lm[indices[:n_train]].reshape(-1, N_LANDMARKS, 2).mean(0)
    adj = build_knn_adj(train_lm_mean, k=8).to(f"cuda:{gpu_ids[0]}")

    # Data loaders
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        args.data_root, all_lm, args.batch_size, num_workers=args.num_workers)

    # Run experiments
    results = {}
    for name in args.experiments:
        if name not in ALL_EXPERIMENTS:
            print(f"[SKIP] Unknown experiment: {name}")
            continue
        model_fn = build_model_fn(name, adj)
        results[name] = run_experiment(
            name=name, model_fn=model_fn,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            epochs=args.epochs, lr=args.lr, label_smoothing=args.label_smooth,
            gpu_ids=gpu_ids, save_dir=args.save_dir)

    # Save all results
    result_path = os.path.join(args.save_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved → {result_path}")

    # Summary
    print("\n" + "="*50)
    print("  SUMMARY")
    print("="*50)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_acc"]):
        print(f"  {name:<22}  Test: {r['test_acc']:.4f}  "
              f"Val: {r['best_val_acc']:.4f}  BestEp: {r['best_epoch']}")


if __name__ == "__main__":
    main()
