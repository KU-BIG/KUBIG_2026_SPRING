#!/usr/bin/env python3
"""Batch face crop using MTCNN with multi-GPU parallelism."""
import os
import torch
import torch.multiprocessing as mp
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
from tqdm import tqdm

SRC = Path("/home/ubuntu/Junyoung/KUBIG/contest/merged_7class_all")
DST = Path("/home/ubuntu/Junyoung/KUBIG/contest/cropped_7class")
TARGET_SIZE = (224, 224)
MARGIN = 20
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def process_chunk(gpu_id, file_list):
    """Process a chunk of files on a specific GPU."""
    device = torch.device(f'cuda:{gpu_id}')
    mtcnn = MTCNN(keep_all=False, device=device, min_face_size=20,
                  thresholds=[0.5, 0.6, 0.6], post_process=False)

    detected = 0
    failed = 0

    for cls, fname in tqdm(file_list, desc=f"GPU{gpu_id}", position=gpu_id):
        src_path = SRC / cls / fname
        dst_path = DST / cls / fname

        if dst_path.exists():
            continue

        try:
            img = Image.open(src_path).convert("RGB")
            boxes, probs = mtcnn.detect(img)

            if boxes is not None and probs[0] > 0.5:
                x1, y1, x2, y2 = boxes[0]
                w, h = img.size
                x1 = max(0, int(x1) - MARGIN)
                y1 = max(0, int(y1) - MARGIN)
                x2 = min(w, int(x2) + MARGIN)
                y2 = min(h, int(y2) + MARGIN)
                img = img.crop((x1, y1, x2, y2))
                detected += 1

            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(dst_path, quality=95)

        except Exception as e:
            try:
                img = Image.open(src_path).convert("RGB")
                img.resize(TARGET_SIZE, Image.LANCZOS).save(dst_path, quality=95)
            except:
                failed += 1

    return detected, failed


def main():
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")

    # Collect all files
    all_files = []
    for cls in CLASSES:
        (DST / cls).mkdir(parents=True, exist_ok=True)
        src_dir = SRC / cls
        for f in os.listdir(src_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                all_files.append((cls, f))

    print(f"Total files: {len(all_files)}")

    # Split across GPUs
    chunks = [[] for _ in range(n_gpus)]
    for i, item in enumerate(all_files):
        chunks[i % n_gpus].append(item)

    # Launch processes
    mp.set_start_method('spawn', force=True)
    with mp.Pool(n_gpus) as pool:
        results = pool.starmap(process_chunk, [(i, chunks[i]) for i in range(n_gpus)])

    total_detected = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)
    total = len(all_files)

    print(f"\n=== DONE ===")
    print(f"Total: {total}")
    print(f"Face detected & cropped: {total_detected} ({total_detected/total*100:.1f}%)")
    print(f"No face (used original): {total - total_detected - total_failed}")
    print(f"Failed: {total_failed}")


if __name__ == "__main__":
    main()
