#!/usr/bin/env python3
"""
Facial Expression Recognition — Inference
==========================================

Load a trained GCN_FiLM_L34 checkpoint and predict emotions from images.
Optionally generates Grad-CAM heatmaps to visualize model attention.

Usage
-----
  # Single image
  python inference.py --checkpoint best_GCN_FiLM_L34.pt --image face.jpg

  # Directory of images
  python inference.py --checkpoint best_GCN_FiLM_L34.pt --image-dir ./test_images/

  # With Grad-CAM visualization
  python inference.py --checkpoint best_GCN_FiLM_L34.pt --image face.jpg --gradcam

  # CPU only
  python inference.py --checkpoint best_GCN_FiLM_L34.pt --image face.jpg --device cpu
"""

import argparse
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# ── Model classes (same architecture as train_fer_gcn_film.py) ──
import pickle

N_LANDMARKS   = 478
LM_DIM        = N_LANDMARKS * 2
GNN_OUT_DIM   = 256
NUM_CLASSES   = 7
CLASS_NAMES   = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_EMOJI   = ["😠", "🤢", "😨", "😄", "😐", "😢", "😲"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
VGGFACE2_PKL  = os.path.join(os.path.dirname(__file__), "pretrained", "resnet50_ft_weight.pkl")


# ══════════════════════════════════════════════════════════════
#  Model Definition
# ══════════════════════════════════════════════════════════════

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
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i+1]) for i in range(n_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers - 1)])

    def forward(self, lm_flat):
        x = lm_flat.view(-1, N_LANDMARKS, 2)
        for i, layer in enumerate(self.layers):
            x = layer(x, self.adj)
            if i < len(self.layers) - 1:
                x = F.relu(self.norms[i](x))
        return x.mean(dim=1)


class FiLM2d(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.ReLU(),
                                      nn.Linear(cond_dim, num_features))
        self.beta_fc  = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.ReLU(),
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


class GCNFiLM_L34(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, adj=None):
        super().__init__()
        self.gnn = LandmarkGCN(adj=adj)
        bb = models.resnet50(weights=None)
        if os.path.exists(VGGFACE2_PKL):
            with open(VGGFACE2_PKL, "rb") as f:
                weights = pickle.load(f, encoding="latin1")
            sd = bb.state_dict()
            for k, v in weights.items():
                t = torch.from_numpy(v)
                if k in sd and sd[k].shape == t.shape:
                    sd[k] = t
            bb.load_state_dict(sd)
        self.stem    = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1  = bb.layer1
        self.layer2  = bb.layer2
        self.layer3  = FiLMLayer([FiLMBottleneck(b, GNN_OUT_DIM) for b in bb.layer3.children()])
        self.layer4  = FiLMLayer([FiLMBottleneck(b, GNN_OUT_DIM) for b in bb.layer4.children()])
        self.avgpool = bb.avgpool
        self.fc      = nn.Linear(2048, num_classes)

    def forward(self, x, lm):
        cond = self.gnn(lm)
        x    = self.stem(x)
        x    = self.layer1(x)
        x    = self.layer2(x)
        x    = self.layer3(x, cond)
        x    = self.layer4(x, cond)
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ══════════════════════════════════════════════════════════════
#  Landmark Extractor
# ══════════════════════════════════════════════════════════════

class LandmarkExtractor:
    """Extracts 478 MediaPipe face landmarks from an image."""

    TASK_URL = ("https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

    def __init__(self, task_path: str = "face_landmarker_v2_with_blendshapes.task"):
        if not os.path.exists(task_path):
            print(f"Downloading MediaPipe model to {task_path} ...")
            urllib.request.urlretrieve(self.TASK_URL, task_path)

        import mediapipe as mp
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=task_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1, min_face_detection_confidence=0.2)
        self._mp       = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def extract(self, pil_image: Image.Image) -> np.ndarray:
        """
        Returns:
            landmarks: (956,) float32 array of flattened (x, y) coords,
                       or zeros if no face detected.
        """
        mp_img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.array(pil_image.convert("RGB")))
        result = self._landmarker.detect(mp_img)
        if result.face_landmarks:
            coords = np.array([[lm.x, lm.y] for lm in result.face_landmarks[0]],
                               dtype=np.float32)
            return coords.flatten()
        return np.zeros(LM_DIM, dtype=np.float32)

    def close(self):
        self._landmarker.close()


# ══════════════════════════════════════════════════════════════
#  Grad-CAM
# ══════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(self, x, lm, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x, lm)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, (224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()


def save_gradcam(img_np, cam, pred_cls, probs, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = 0.55 * img_np + 0.45 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np);    axes[0].set_title("Original");      axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay);   axes[2].set_title(f"Overlay\nPred: {CLASS_NAMES[pred_cls]} "
                                                  f"({probs[pred_cls]:.1%})"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM saved → {save_path}")


# ══════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> GCNFiLM_L34:
    state = torch.load(checkpoint_path, map_location=device)
    adj_key = next((k for k in state if "adj" in k), None)
    adj = state[adj_key].to(device) if adj_key else None
    model = GCNFiLM_L34(adj=adj).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def predict(model, image_path: str, extractor: LandmarkExtractor,
            device: torch.device, gradcam: bool = False, save_dir: str = "."):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img     = Image.open(image_path).convert("RGB")
    lm      = extractor.extract(img)
    x       = transform(img).unsqueeze(0).to(device)
    lm_t    = torch.from_numpy(lm).float().unsqueeze(0).to(device)
    img_224 = np.array(img.resize((224, 224))) / 255.0

    if gradcam:
        x.requires_grad_(True)
        cam_fn  = GradCAM(model, model.layer4.blocks[-1])
        cam, pred_cls, probs = cam_fn(x, lm_t)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_gradcam(img_224, cam, pred_cls, probs,
                     os.path.join(save_dir, f"{stem}_gradcam.png"))
    else:
        with torch.no_grad():
            logits = model(x, lm_t)
            probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_cls = probs.argmax()

    return pred_cls, probs


def print_result(image_path: str, pred_cls: int, probs: np.ndarray):
    print(f"\n{'─'*45}")
    print(f"  Image : {os.path.basename(image_path)}")
    print(f"  Prediction : {CLASS_EMOJI[pred_cls]} {CLASS_NAMES[pred_cls].upper()}  "
          f"({probs[pred_cls]:.1%})")
    print(f"{'─'*45}")
    print("  All probabilities:")
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar   = "█" * int(prob * 30)
        mark  = " ◀" if i == pred_cls else ""
        print(f"  {CLASS_EMOJI[i]} {cls:<10} {prob:5.1%}  {bar}{mark}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FER Inference — GCN_FiLM_L34")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     type=str, help="Path to a single image")
    group.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--landmark-model", type=str,
                        default="face_landmarker_v2_with_blendshapes.task",
                        help="Path to MediaPipe face landmarker task file")
    parser.add_argument("--gradcam",  action="store_true",
                        help="Generate Grad-CAM heatmap visualizations")
    parser.add_argument("--save-dir", type=str, default="./inference_results",
                        help="Directory to save Grad-CAM outputs")
    parser.add_argument("--device",   type=str, default="cuda",
                        help="Device: cuda / cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.gradcam:
        os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)

    # Initialize landmark extractor
    extractor = LandmarkExtractor(task_path=args.landmark_model)

    # Collect image paths
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in sorted(os.listdir(args.image_dir))
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]
        print(f"Found {len(image_paths)} images in {args.image_dir}")

    # Run inference
    for img_path in image_paths:
        try:
            pred_cls, probs = predict(
                model, img_path, extractor, device,
                gradcam=args.gradcam, save_dir=args.save_dir)
            print_result(img_path, pred_cls, probs)
        except Exception as e:
            print(f"  [ERROR] {img_path}: {e}")

    extractor.close()


if __name__ == "__main__":
    main()
