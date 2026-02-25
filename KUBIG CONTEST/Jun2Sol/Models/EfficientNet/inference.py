"""
EfficientNet-B2 FER Inference

    python inference.py --checkpoint ./results/best_final.pt --image face.jpg
    python inference.py --checkpoint ./results/best_phase2.pt --image face.jpg --topk 5
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image


EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# ---------------------------------------------------------------------------
# Model  (must match train_fer.py / train_fer_final.py)
# ---------------------------------------------------------------------------

class EfficientNetFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone   = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(ckpt_path: str, device: torch.device) -> EfficientNetFER:
    model = EfficientNetFER(num_classes=len(EMOTIONS))
    raw   = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        state = raw.get('model', raw.get('state_dict', raw))
    else:
        state = raw
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def predict(model: nn.Module, image_path: str, device: torch.device) -> torch.Tensor:
    img    = Image.open(image_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return probs.cpu()


def print_results(probs: torch.Tensor, image_path: str, topk: int = 3):
    topk = min(topk, len(EMOTIONS))
    top_probs, top_idx = probs.topk(topk)

    print(f'\nImage : {image_path}')
    print(f'Result: {EMOTIONS[top_idx[0].item()].upper()}  ({top_probs[0].item()*100:.1f}%)')
    print(f'\nTop-{topk} predictions:')
    for i in range(topk):
        emotion = EMOTIONS[top_idx[i].item()]
        conf    = top_probs[i].item()
        bar     = '█' * int(conf * 30)
        print(f'  {emotion:>10s}  {conf*100:5.1f}%  {bar}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='EfficientNet-B2 FER inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint (best_final.pt or best_phase2.pt)')
    parser.add_argument('--image',      type=str, required=True,
                        help='Input face image path')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topk',       type=int, default=3,
                        help='Number of top predictions to display')
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    if not Path(args.image).exists():
        raise FileNotFoundError(f'Image not found: {args.image}')

    device = torch.device(args.device)
    print(f'Device: {device}')

    model = load_model(args.checkpoint, device)
    print(f'Loaded: {args.checkpoint}')

    probs = predict(model, args.image, device)
    print_results(probs, args.image, topk=args.topk)


if __name__ == '__main__':
    main()
