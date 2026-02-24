#!/usr/bin/env python3
"""
Dataset cleaning: perceptual hash dedup + face confidence filtering.

Step 1: Perceptual hashing (pHash) to find near-duplicate images.
  - Same class duplicates: remove extras (keep one).
  - Cross-class duplicates: remove all copies (ambiguous label).
Step 2: MTCNN face confidence scoring on cropped images.
  - Remove images where confidence < threshold (default 0.9).
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image
import imagehash
import torch
import torch.multiprocessing as mp
from facenet_pytorch import MTCNN
from tqdm import tqdm

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ── Step 1: Perceptual Hashing ─────────────────────────────────────────────

def compute_hashes(data_root, hash_size=16):
    """Compute pHash for every image. Returns {hash_str: [(cls, fname), ...]}."""
    data_root = Path(data_root)
    hash_map = defaultdict(list)  # hash -> [(cls, fname)]
    total = sum(len(os.listdir(data_root / c)) for c in CLASSES)

    with tqdm(total=total, desc="Computing pHash") as pbar:
        for cls in CLASSES:
            cls_dir = data_root / cls
            for fname in sorted(os.listdir(cls_dir)):
                fpath = cls_dir / fname
                try:
                    img = Image.open(fpath).convert("RGB")
                    h = str(imagehash.phash(img, hash_size=hash_size))
                    hash_map[h].append((cls, fname))
                except Exception as e:
                    print(f"  [hash error] {cls}/{fname}: {e}")
                pbar.update(1)

    return hash_map


class BKTree:
    """BK-tree for efficient hamming distance nearest neighbor search."""

    def __init__(self, distance_fn):
        self.distance_fn = distance_fn
        self.root = None

    def insert(self, item):
        if self.root is None:
            self.root = (item, {})
            return
        node = self.root
        while True:
            d = self.distance_fn(node[0], item)
            if d == 0:
                return  # duplicate
            if d in node[1]:
                node = node[1][d]
            else:
                node[1][d] = (item, {})
                return

    def query(self, item, threshold):
        """Find all items within hamming distance threshold."""
        if self.root is None:
            return []
        results = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = self.distance_fn(node[0], item)
            if d <= threshold:
                results.append((node[0], d))
            for dist, child in node[1].items():
                if d - threshold <= dist <= d + threshold:
                    stack.append(child)
        return results


def find_duplicates(hash_map, hamming_threshold=6):
    """
    From exact-match hash groups, find:
    - same_class_dups: same hash, same class → keep one, remove rest
    - cross_class_dups: same hash, different classes → remove all (ambiguous)

    Also does BK-tree based near-duplicate detection (fast).
    """
    same_class_remove = []
    cross_class_remove = []

    # --- Exact duplicates (same hash) ---
    for h, entries in hash_map.items():
        if len(entries) <= 1:
            continue
        classes_in_group = set(e[0] for e in entries)
        if len(classes_in_group) == 1:
            same_class_remove.extend(entries[1:])
        else:
            cross_class_remove.extend(entries)

    # --- Near-duplicates via BK-tree ---
    hash_list = list(hash_map.keys())
    hash_objs = {h: imagehash.hex_to_hash(h) for h in hash_list}

    def hamming_dist(h1, h2):
        return hash_objs[h1] - hash_objs[h2]

    print(f"  Building BK-tree with {len(hash_list)} unique hashes...")
    tree = BKTree(hamming_dist)
    for h in tqdm(hash_list, desc="Building BK-tree"):
        tree.insert(h)

    merged = set()  # hashes already grouped
    for h1 in tqdm(hash_list, desc="Near-dup query"):
        if h1 in merged:
            continue
        neighbors = tree.query(h1, hamming_threshold)
        # Only consider neighbors that are different hashes (dist > 0)
        near_hashes = [nh for nh, d in neighbors if d > 0 and nh not in merged]
        if not near_hashes:
            continue
        # Group: h1 + all its near-dup neighbors
        group_entries = list(hash_map[h1])
        for nh in near_hashes:
            group_entries.extend(hash_map[nh])
            merged.add(nh)
        merged.add(h1)

        classes_in_group = set(e[0] for e in group_entries)
        if len(classes_in_group) == 1:
            # Same class near-dup: keep first, remove rest
            same_class_remove.extend(group_entries[1:])
        else:
            # Cross-class near-dup: remove all
            cross_class_remove.extend(group_entries)

    return same_class_remove, cross_class_remove


# ── Step 2: Face Confidence ────────────────────────────────────────────────

def compute_confidence_chunk(gpu_id, file_list, data_root):
    """Compute MTCNN face confidence for a chunk of files on a specific GPU."""
    device = torch.device(f'cuda:{gpu_id}')
    mtcnn = MTCNN(keep_all=False, device=device, min_face_size=20,
                  thresholds=[0.3, 0.4, 0.4], post_process=False)

    results = {}  # (cls, fname) -> confidence
    for cls, fname in tqdm(file_list, desc=f"Conf GPU{gpu_id}", position=gpu_id):
        fpath = Path(data_root) / cls / fname
        try:
            img = Image.open(fpath).convert("RGB")
            boxes, probs = mtcnn.detect(img)
            if boxes is not None and probs is not None:
                results[(cls, fname)] = float(probs[0])
            else:
                results[(cls, fname)] = 0.0
        except Exception:
            results[(cls, fname)] = 0.0
    return results


def compute_confidence_parallel(data_root, n_gpus=None):
    """Compute face confidence for all images using multiple GPUs."""
    data_root = Path(data_root)
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()

    all_files = []
    for cls in CLASSES:
        for fname in sorted(os.listdir(data_root / cls)):
            all_files.append((cls, fname))

    chunks = [[] for _ in range(n_gpus)]
    for i, item in enumerate(all_files):
        chunks[i % n_gpus].append(item)

    mp.set_start_method('spawn', force=True)
    with mp.Pool(n_gpus) as pool:
        chunk_results = pool.starmap(
            compute_confidence_chunk,
            [(i, chunks[i], str(data_root)) for i in range(n_gpus)]
        )

    # Merge results
    all_conf = {}
    for cr in chunk_results:
        all_conf.update(cr)
    return all_conf


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./cropped_7class")
    parser.add_argument("--output", default="./cleaned_7class")
    parser.add_argument("--hash-size", type=int, default=16)
    parser.add_argument("--hamming-thresh", type=int, default=6,
                        help="Hamming distance threshold for near-dup (default: 6)")
    parser.add_argument("--conf-thresh", type=float, default=0.9,
                        help="Minimum MTCNN face confidence (default: 0.9)")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report stats, don't copy files")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output = Path(args.output)

    # Count originals
    orig_counts = {}
    for cls in CLASSES:
        orig_counts[cls] = len(os.listdir(data_root / cls))
    total_orig = sum(orig_counts.values())
    print(f"\n=== Original dataset: {total_orig} images ===")
    for cls in CLASSES:
        print(f"  {cls}: {orig_counts[cls]}")

    # Step 1: Perceptual hashing
    print(f"\n{'='*60}")
    print("Step 1: Perceptual Hashing (dedup)")
    print(f"{'='*60}")
    hash_map = compute_hashes(data_root, args.hash_size)
    same_remove, cross_remove = find_duplicates(hash_map, args.hamming_thresh)

    dup_remove_set = set()
    for cls, fname in same_remove:
        dup_remove_set.add((cls, fname))
    for cls, fname in cross_remove:
        dup_remove_set.add((cls, fname))

    print(f"\n  Same-class duplicates to remove: {len(same_remove)}")
    print(f"  Cross-class duplicates to remove: {len(cross_remove)}")
    print(f"  Total phash removals: {len(dup_remove_set)}")

    # Step 2: Face confidence
    print(f"\n{'='*60}")
    print("Step 2: Face Confidence Filtering")
    print(f"{'='*60}")
    all_conf = compute_confidence_parallel(data_root, args.gpus)

    low_conf_set = set()
    conf_values = sorted(all_conf.values())
    print(f"\n  Confidence distribution:")
    for thresh in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
        count = sum(1 for v in conf_values if v >= thresh)
        print(f"    >= {thresh:.2f}: {count} ({count/len(conf_values)*100:.1f}%)")

    for (cls, fname), conf in all_conf.items():
        if conf < args.conf_thresh:
            low_conf_set.add((cls, fname))
    print(f"\n  Below {args.conf_thresh} threshold: {len(low_conf_set)}")

    # Combine removals
    total_remove = dup_remove_set | low_conf_set
    both = dup_remove_set & low_conf_set
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Phash removals:      {len(dup_remove_set)}")
    print(f"  Low-conf removals:   {len(low_conf_set)}")
    print(f"  Overlap:             {len(both)}")
    print(f"  Total to remove:     {len(total_remove)}")
    print(f"  Remaining:           {total_orig - len(total_remove)}")

    # Per-class breakdown
    print(f"\n  Per-class breakdown:")
    for cls in CLASSES:
        cls_orig = orig_counts[cls]
        cls_remove = sum(1 for c, f in total_remove if c == cls)
        cls_remain = cls_orig - cls_remove
        print(f"    {cls:10s}: {cls_orig} -> {cls_remain} (removed {cls_remove})")

    if args.dry_run:
        print("\n[DRY RUN] No files copied.")
        # Save removal list for inspection
        report = {
            "same_class_dups": [(c, f) for c, f in same_remove],
            "cross_class_dups": [(c, f) for c, f in cross_remove],
            "low_confidence": [(c, f, all_conf.get((c, f), 0)) for c, f in low_conf_set],
        }
        report_path = data_root.parent / "cleaning_report.json"
        with open(report_path, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"Report saved to {report_path}")
        return

    # Copy clean files to output
    print(f"\nCopying clean files to {output} ...")
    for cls in CLASSES:
        (output / cls).mkdir(parents=True, exist_ok=True)

    copied = 0
    for cls in CLASSES:
        cls_dir = data_root / cls
        for fname in tqdm(sorted(os.listdir(cls_dir)), desc=cls):
            if (cls, fname) in total_remove:
                continue
            src = cls_dir / fname
            dst = output / cls / fname
            shutil.copy2(src, dst)
            copied += 1

    print(f"\n=== Cleaning Complete ===")
    print(f"  Copied {copied} clean images to {output}")


if __name__ == "__main__":
    main()
