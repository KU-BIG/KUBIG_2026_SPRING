"""
EfficientNet-B2 FER Training (Phase 1 & 2)

Phase 1: Classification head only
    python train_fer.py --data-root ./cleaned_7class --gpus 0,1 --pretrained affectnet --phase 1

Phase 2: Full fine-tuning
    python train_fer.py --data-root ./cleaned_7class --gpus 0,1 --pretrained affectnet --phase 2 --lr 1e-4
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

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def load_affectnet_weights(model):
    ckpt_path = Path('pretrained/efficientnet_b2_affectnet.pth')
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location='cpu')
        state = state.get('state_dict', state)
        backbone_state = {k.replace('backbone.', ''): v
                          for k, v in state.items() if k.startswith('backbone.')}
        if backbone_state:
            model.backbone.load_state_dict(backbone_state, strict=False)
        else:
            model.backbone.load_state_dict(state, strict=False)
        print(f'Loaded AffectNet weights from {ckpt_path}')
    else:
        print('AffectNet weights not found — falling back to ImageNet pretrained.')
        imagenet = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
        model.backbone.load_state_dict(imagenet.state_dict())


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
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_correct += (out.argmax(1) == labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_correct += (out.argmax(1) == labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

    return total_loss / total_n, total_correct / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',  type=str, required=True)
    parser.add_argument('--gpus',       type=str, default='0')
    parser.add_argument('--pretrained', type=str, default='imagenet',
                        choices=['imagenet', 'affectnet'])
    parser.add_argument('--phase',      type=int, default=1, choices=[1, 2])
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers',type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--resume',     type=str, default=None)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(',')]
    device  = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    # ---- transforms -------------------------------------------------------
    train_tf = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
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

    # ---- model ------------------------------------------------------------
    model = EfficientNetFER(num_classes=7)

    if args.pretrained == 'affectnet':
        load_affectnet_weights(model)

    if args.phase == 1:
        print('Phase 1: training classification head only')
        model.freeze_backbone()
        lr = args.lr
        param_groups = model.classifier.parameters()
    else:
        print('Phase 2: fine-tuning entire network')
        model.unfreeze_backbone()
        lr = args.lr if args.lr != 1e-3 else 1e-4
        # Auto-load phase-1 best unless --resume provided
        phase1_ckpt = Path(args.output_dir) / 'best_phase1.pt'
        if args.resume is None and phase1_ckpt.exists():
            state = torch.load(phase1_ckpt, map_location='cpu')
            model.load_state_dict(state['model'])
            print(f'Loaded Phase-1 checkpoint from {phase1_ckpt}')
        param_groups = model.parameters()

    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state['model'])
        print(f'Resumed from {args.resume}')

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    # ---- training loop ----------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()
    os.makedirs(args.output_dir, exist_ok=True)
    best_acc  = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler)
        vl_loss, vl_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        print(f'[Phase {args.phase}] Epoch {epoch:03d}/{args.epochs} | '
              f'Train {tr_loss:.4f}/{tr_acc:.4f} | Val {vl_loss:.4f}/{vl_acc:.4f}')

        if vl_acc > best_acc:
            best_acc = vl_acc
            m = model.module if hasattr(model, 'module') else model
            torch.save({'model': m.state_dict(), 'epoch': epoch, 'val_acc': vl_acc},
                       Path(args.output_dir) / f'best_phase{args.phase}.pt')
            print(f'  Saved (val_acc={vl_acc:.4f})')

    print(f'Done. Best val acc: {best_acc:.4f}')


if __name__ == '__main__':
    main()
