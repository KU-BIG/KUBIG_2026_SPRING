"""
EfficientNet-B2 FER — Final Training
Supports Weighted Cross Entropy and Label Smoothing.

    python train_fer_final.py --data-root ./cleaned_7class --gpus 0,1 \
        --loss weighted_ce --use-label-smoothing
"""

import argparse
import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import timm
from PIL import Image


EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FERDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.samples = []

        for label_idx, emotion in enumerate(EMOTIONS):
            emotion_dir = self.root / emotion
            if not emotion_dir.exists():
                continue
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for img_path in emotion_dir.glob(ext):
                    self.samples.append((str(img_path), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_counts(self):
        counts = Counter(label for _, label in self.samples)
        return [counts.get(i, 0) for i in range(len(EMOTIONS))]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EfficientNetFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_p = nn.functional.log_softmax(pred, dim=1)
        nll   = -log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -log_p.mean(dim=1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


class WeightedLabelSmoothingCE(nn.Module):
    def __init__(self, weights, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        log_p  = nn.functional.log_softmax(pred, dim=1)
        nll    = -log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -log_p.mean(dim=1)
        loss   = (1 - self.smoothing) * nll + self.smoothing * smooth
        w      = self.weights[target]
        return (loss * w).sum() / w.sum()


def build_criterion(args, class_counts, device):
    counts  = torch.tensor(class_counts, dtype=torch.float32)
    weights = (counts.sum() / (len(counts) * counts.clamp(min=1)))
    weights = (weights / weights.sum()).to(device)

    if args.loss == 'weighted_ce':
        if args.use_label_smoothing:
            print('Loss: Weighted Cross Entropy + Label Smoothing (0.1)')
            return WeightedLabelSmoothingCE(weights, smoothing=0.1)
        else:
            print('Loss: Weighted Cross Entropy')
            return nn.CrossEntropyLoss(weight=weights)
    else:  # plain ce
        if args.use_label_smoothing:
            print('Loss: Cross Entropy + Label Smoothing (0.1)')
            return LabelSmoothingCE(smoothing=0.1)
        else:
            print('Loss: Cross Entropy')
            return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = total_correct = total_n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_correct += (out.argmax(1) == labels).sum().item()
        total_loss    += loss.item() * imgs.size(0)
        total_n       += imgs.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_correct += (out.argmax(1) == labels).sum().item()
        total_loss    += loss.item() * imgs.size(0)
        total_n       += imgs.size(0)

    return total_loss / total_n, total_correct / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',           type=str,   required=True)
    parser.add_argument('--gpus',                type=str,   default='0')
    parser.add_argument('--loss',                type=str,   default='weighted_ce',
                        choices=['ce', 'weighted_ce'])
    parser.add_argument('--use-label-smoothing', action='store_true')
    parser.add_argument('--epochs',              type=int,   default=50)
    parser.add_argument('--lr',                  type=float, default=5e-5)
    parser.add_argument('--batch-size',          type=int,   default=64)
    parser.add_argument('--num-workers',         type=int,   default=4)
    parser.add_argument('--output-dir',          type=str,   default='./results')
    parser.add_argument('--checkpoint',          type=str,   default=None,
                        help='Starting checkpoint (default: best_phase2.pt)')
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(',')]
    device  = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    # ---- transforms -------------------------------------------------------
    train_tf = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FERDataset(args.data_root, 'train', train_tf)
    val_ds   = FERDataset(args.data_root, 'val',   val_tf)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')
    class_counts = train_ds.get_class_counts()
    print(f'Class counts: {dict(zip(EMOTIONS, class_counts))}')

    # ---- model ------------------------------------------------------------
    model = EfficientNetFER(num_classes=7)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(args.output_dir) / 'best_phase2.pt'
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state.get('model', state), strict=False)
        print(f'Loaded checkpoint from {ckpt_path}')
    else:
        print(f'No checkpoint found at {ckpt_path}, training from scratch.')

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    # ---- training loop ----------------------------------------------------
    criterion = build_criterion(args, class_counts, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10,
        steps_per_epoch=len(train_dl), epochs=args.epochs,
    )
    scaler   = GradScaler()
    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler)
        vl_loss, vl_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch:03d}/{args.epochs} | '
              f'Train {tr_loss:.4f}/{tr_acc:.4f} | Val {vl_loss:.4f}/{vl_acc:.4f}')

        if vl_acc > best_acc:
            best_acc = vl_acc
            m = model.module if hasattr(model, 'module') else model
            torch.save(
                {'model': m.state_dict(), 'epoch': epoch,
                 'val_acc': vl_acc, 'args': vars(args)},
                Path(args.output_dir) / 'best_final.pt',
            )
            print(f'  Saved (val_acc={vl_acc:.4f})')

    print(f'Done. Best val acc: {best_acc:.4f}')


if __name__ == '__main__':
    main()
