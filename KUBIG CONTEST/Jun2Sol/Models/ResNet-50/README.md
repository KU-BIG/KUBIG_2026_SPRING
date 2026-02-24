# FER with GCN Landmark Encoder + FiLM Conditioning

Facial Expression Recognition (FER) using a **ResNet-50** backbone conditioned by **GCN-encoded facial landmarks** via **FiLM (Feature-wise Linear Modulation)**.

**Best model (GCN_FiLM_L34): Test Accuracy 87.12%**

---

## Overview

Standard FER models rely solely on image features. This work augments a face-pretrained ResNet-50 with structural information from 478 MediaPipe facial landmarks encoded by a Graph Convolutional Network (GCN), injected via FiLM conditioning at ResNet layers 3 and 4.

```
Input image  ──► ResNet-50 (VGGFace2)
                     ▲ FiLM at layer3, layer4
Landmarks    ──► GCN encoder ──► 256-dim embedding
```

### Key Results

| Model | Test Acc |
|---|---|
| Baseline (ResNet-50 + VGGFace2) | 86.18% |
| **GCN_FiLM_L34 (ours)** | **87.12%** |
| GCN_FiLM_L4 | 86.72% |
| GCN_FiLM_L234 | 86.91% |
| GAT_FiLM_L34 | 86.54% |

#### Per-Class Accuracy (GCN_FiLM_L34)

| Class | Accuracy | Samples |
|---|---|---|
| happy | 95.4% | 1,206 |
| surprise | 88.0% | 485 |
| neutral | 87.9% | 900 |
| sad | 86.6% | 821 |
| angry | 83.4% | 428 |
| fear | 58.5% | 123 |
| disgust | 54.2% | 168 |

> disgust/fear performance is limited by class imbalance. Using class-weighted loss improves disgust to 61.9% (+7.7%) and fear to 60.2% (+1.7%) at a small overall accuracy cost (86.37%).

---

## Method

### GCN Landmark Encoder
- **Input**: 478 MediaPipe face landmarks → flattened (956-dim)
- **Graph**: k-NN (k=8) adjacency built from mean landmark positions
- **Architecture**: 3-layer GCN → LayerNorm → ReLU → global mean pool → 256-dim
- **Output**: structural embedding encoding facial geometry

### FiLM Conditioning
- **What**: Feature-wise Linear Modulation applies per-channel scale/shift to CNN feature maps
- **How**: `FiLM(x, c) = γ(c) · x + β(c)` where γ, β are MLPs of the GCN embedding
- **Where**: Applied after BatchNorm in each Bottleneck block of the conditioned layers

### Ablation Study
We compare injection at different ResNet stages:
- `L4`: layer4 only
- `L34`: layer3 + layer4 ← **best**
- `L234`: layer2 + layer3 + layer4
- `All`: all layers including stem

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/fer-gcn-film.git
cd fer-gcn-film
pip install -r requirements.txt
```

#### VGGFace2 Pretrained Weights (required)
Download from [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) and place at:
```
pretrained/resnet50_ft_weight.pkl
```

#### Dataset
We use a cleaned 7-class FER dataset (RAF-DB based, 33,000+ images after cleaning).
Prepare your dataset as:
```
cleaned_7class/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

---

## Training

### Full Ablation Study (GCN_FiLM variants)
```bash
# All 9 experiments on 8 GPUs (one per GPU)
python train_fer_gcn_film.py \
    --data-root ./cleaned_7class \
    --save-dir  ./results \
    --gpus      0,1,2,3,4,5,6,7

# Best model only (GCN_FiLM_L34) on 2 GPUs
python train_fer_gcn_film.py \
    --data-root  ./cleaned_7class \
    --gpus       0,1 \
    --experiments GCN_FiLM_L34
```

### Class-Weighted Training (Minority Class Improvement)
```bash
python train_fer_cls_weight.py \
    --data-root ./cleaned_7class \
    --gpus      0,1
```

---

## Inference

```bash
# Single image
python inference.py \
    --checkpoint ./results/best_GCN_FiLM_L34.pt \
    --image      face.jpg

# Directory of images
python inference.py \
    --checkpoint  ./results/best_GCN_FiLM_L34.pt \
    --image-dir   ./test_images/

# With Grad-CAM visualization
python inference.py \
    --checkpoint ./results/best_GCN_FiLM_L34.pt \
    --image      face.jpg \
    --gradcam    \
    --save-dir   ./output/
```

Example output:
```
─────────────────────────────────────────────
  Image      : face.jpg
  Prediction : 😄 HAPPY  (96.3%)
─────────────────────────────────────────────
  All probabilities:
  😠 angry      1.2%  ███
  🤢 disgust    0.3%  █
  😨 fear       0.5%  █
  😄 happy     96.3%  ██████████████████████████████ ◀
  😐 neutral    1.1%  ███
  😢 sad        0.3%  █
  😲 surprise   0.3%  █
```

---

## File Structure

```
fer-gcn-film/
├── train_fer_gcn_film.py    # GCN_FiLM ablation study training
├── train_fer_cls_weight.py  # Class-weighted loss training (minority improvement)
├── inference.py             # Inference + Grad-CAM visualization
├── requirements.txt
└── results/
    ├── graph_v14.png            # V14 FiLM ablation results
    ├── graph_v15_gcn.png        # V15 GCN_FiLM results
    ├── graph_v15_gat.png        # V15 GAT_FiLM results
    ├── graph_v15_perclass.png   # Per-class accuracy
    └── graph_v15_vs_v18.png     # Minority class improvement comparison
```

---

## Grad-CAM Analysis

Grad-CAM comparison shows that:
- **Baseline** focuses on nose/eye regions (identity bias from VGGFace2)
- **GCN_FiLM** distributes attention to expression-relevant regions (mouth corners, eyebrows) — evidence that landmark conditioning steers the model toward facial geometry

---

## Requirements

- Python 3.8+
- PyTorch 1.13+
- CUDA GPU (recommended: A100 / V100)
- MediaPipe 0.10+
