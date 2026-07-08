#!/usr/bin/env python3
"""Grid search over HAN + head/training hyperparameters for the frozen-PDGNN + HAN
node-fusion fine-tuning model.

The PDGNN backbone checkpoint is FIXED for every run (the "frozen foundation"),
so this sweep only tunes the trainable part: the HAN branch (hidden dim, layers,
heads, dropout) plus the head/optimizer knobs (lr, dropout, weight_decay).

Each (combo, seed) is trained via train/train_pdgnn_han_finetune.py and its result
JSON is written to results/sweep/<config>/ so runs never clobber each other.

Because MolHIV ROC-AUC is high-variance, pass multiple seeds (--seeds 0,1,2) and
combos are ranked by MEAN valid ROC-AUC over seeds (with std shown), i.e. by
stability rather than a single lucky run.

Example:
    python -u scripts/sweep_pdgnn_han_finetune.py \
        --backbone-ckpt results/pdgnn_tda_3d_elec_best.pt \
        --lrs 1e-4,3e-4 --dropouts 0.3,0.5 --weight-decays 1e-5 \
        --han-hiddens 128,256 --han-layers 2 --han-heads 4 --han-dropouts 0.2 \
        --seeds 0,1,2 --epochs 50 --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def _floats(csv: str) -> list[float]:
    return [float(x) for x in csv.split(",") if x.strip()]


def _ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def _bools(csv: str) -> list[bool]:
    return [x.strip() in ("1", "true", "True", "yes") for x in csv.split(",") if x.strip()]


def _combo_tag(lr, do, wd, hh, hl, hd, hdo, bt) -> str:
    return (f"lr{lr:g}_do{do:g}_wd{wd:g}"
            f"_hh{hh}_hl{hl}_hd{hd}_hdo{hdo:g}_bt{int(bt)}")


def _run_pool(jobs, max_parallel: int):
    """Run (cmd, log_path) jobs with at most max_parallel concurrent subprocesses.

    Each job's stdout/stderr is redirected to its own log file so concurrent runs
    don't interleave. Raises RuntimeError if any job exits non-zero.
    """
    pending = list(jobs)
    running = []  # list of (Popen, log_file_handle, cmd, log_path)
    failures = []
    total = len(pending)
    started = 0

    while pending or running:
        while pending and len(running) < max_parallel:
            cmd, log_path = pending.pop(0)
            started += 1
            print(f"[start {started}/{total}] -> {Path(log_path).name}", flush=True)
            fh = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT)
            running.append((proc, fh, cmd, log_path))

        time.sleep(3)
        still = []
        for proc, fh, cmd, log_path in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, fh, cmd, log_path))
                continue
            fh.close()
            if ret != 0:
                failures.append((cmd, log_path, ret))
                print(f"[FAIL rc={ret}] see {log_path}", flush=True)
            else:
                print(f"[done] {Path(log_path).name}", flush=True)
        running = still

    if failures:
        raise RuntimeError(
            f"{len(failures)} run(s) failed: "
            + "; ".join(f"{Path(lp).name}(rc={rc})" for _, lp, rc in failures)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pdgnn_han_finetune_3d_elec_multifilt")
    parser.add_argument("--backbone-ckpt", type=str, required=True,
                        help="Fixed frozen PDGNN checkpoint used by every run.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=_ints, default=[0],
                        help="Comma-separated seeds, e.g. 0,1,2 (multi-seed = stability).")
    parser.add_argument("--lrs", type=_floats, default=[1e-4, 3e-4])
    parser.add_argument("--dropouts", type=_floats, default=[0.3, 0.5],
                        help="Head dropout rates.")
    parser.add_argument("--weight-decays", type=_floats, default=[1e-5])
    parser.add_argument("--han-hiddens", type=_ints, default=[128, 256])
    parser.add_argument("--han-layers", type=_ints, default=[2])
    parser.add_argument("--han-heads", type=_ints, default=[4])
    parser.add_argument("--han-dropouts", type=_floats, default=[0.2])
    parser.add_argument("--balanced-trains", type=_bools, default=[False],
                        help="Comma-separated 0/1: compare imbalanced (0) vs 1:1 balanced (1) training.")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Number of runs to execute concurrently on the GPU.")
    parser.add_argument("--per-run-workers", type=int, default=2,
                        help="DataLoader workers per run (keep low when running in parallel).")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run even if a result JSON already exists.")
    args = parser.parse_args()

    if not Path(args.backbone_ckpt).exists():
        raise FileNotFoundError(f"Missing backbone checkpoint {args.backbone_ckpt}.")

    sweep_dir = RESULTS_ROOT / "sweep" / f"{args.config}_hantune"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(
        args.lrs, args.dropouts, args.weight_decays,
        args.han_hiddens, args.han_layers, args.han_heads, args.han_dropouts,
        args.balanced_trains,
    ))
    total = len(combos) * len(args.seeds)
    print(f"Sweeping {len(combos)} combos x {len(args.seeds)} seeds = {total} runs "
          f"(config={args.config}, epochs={args.epochs})")

    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build job list (skipping already-computed runs) then execute via a pool.
    # combo_tag -> {"params": {...}, "outs": [out paths per seed]}
    aggregated: dict[str, dict] = {}
    jobs = []  # (cmd, log_path)
    for combo in combos:
        lr, do, wd, hh, hl, hd, hdo, bt = combo
        tag = _combo_tag(*combo)
        aggregated[tag] = {"params": dict(
            lr=lr, dropout=do, weight_decay=wd,
            han_hidden=hh, han_layers=hl, han_heads=hd, han_dropout=hdo,
            balanced_train=bt,
        ), "outs": []}
        for seed in args.seeds:
            out = sweep_dir / f"{tag}_s{seed}.json"
            aggregated[tag]["outs"].append(out)
            if out.exists() and not args.rerun:
                print(f"Skipping {out.name} (exists)")
                continue
            cmd = [
                sys.executable, "-u",
                str(PROJECT_ROOT / "train" / "train_pdgnn_han_finetune.py"),
                "--config", args.config,
                "--backbone-ckpt", args.backbone_ckpt,
                "--lr", str(lr),
                "--dropout", str(do),
                "--weight-decay", str(wd),
                "--han-hidden", str(hh),
                "--han-layers", str(hl),
                "--han-heads", str(hd),
                "--han-dropout", str(hdo),
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--seed", str(seed),
                "--num-workers", str(args.per_run_workers),
                "--out", str(out),
            ]
            if bt:
                cmd.append("--balanced-train")
            jobs.append((cmd, str(logs_dir / f"{tag}_s{seed}.log")))

    print(f"{len(jobs)} run(s) to execute, {total - len(jobs)} already cached; "
          f"max_parallel={args.max_parallel}")
    if jobs:
        _run_pool(jobs, max(1, args.max_parallel))

    # Aggregate every combo's per-seed result JSONs.
    for tag, info in aggregated.items():
        info["runs"] = []
        for out in info["outs"]:
            with open(out, encoding="utf-8") as f:
                info["runs"].append(json.load(f))

    rows = []
    for tag, info in aggregated.items():
        valids = [r.get("valid_rocauc", float("nan")) for r in info["runs"]]
        tests = [r.get("test_rocauc", float("nan")) for r in info["runs"]]
        n = len(valids)
        rows.append({
            "tag": tag,
            **info["params"],
            "n_seeds": n,
            "valid_mean": statistics.fmean(valids),
            "valid_std": statistics.pstdev(valids) if n > 1 else 0.0,
            "test_mean": statistics.fmean(tests),
            "test_std": statistics.pstdev(tests) if n > 1 else 0.0,
        })

    rows.sort(key=lambda r: r["valid_mean"], reverse=True)
    print("\n| lr | head do | wd | HAN h | L | heads | HAN do | 1:1 train | seeds "
          "| valid mean±std | test mean±std |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['lr']:g} | {r['dropout']:g} | {r['weight_decay']:g} "
              f"| {r['han_hidden']} | {r['han_layers']} | {r['han_heads']} "
              f"| {r['han_dropout']:g} | {'yes' if r['balanced_train'] else 'no'} "
              f"| {r['n_seeds']} "
              f"| {r['valid_mean']:.4f}±{r['valid_std']:.4f} "
              f"| {r['test_mean']:.4f}±{r['test_std']:.4f} |")

    if rows:
        best = rows[0]
        print(f"\nBest (by mean valid): {best['tag']} -> "
              f"valid {best['valid_mean']:.4f}±{best['valid_std']:.4f}, "
              f"test {best['test_mean']:.4f}±{best['test_std']:.4f}")
        summary = sweep_dir / "_summary.json"
        with open(summary, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Summary saved to {summary}")


if __name__ == "__main__":
    main()
