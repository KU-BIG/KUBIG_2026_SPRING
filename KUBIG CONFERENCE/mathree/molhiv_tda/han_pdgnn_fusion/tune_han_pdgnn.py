#!/usr/bin/env python3
"""Random-search hyperparameter tuning for HAN+PDGNN fusion (main model)."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_han_pdgnn import run_training  # noqa: E402
from utils import load_config  # noqa: E402


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_search_space(path: Path) -> dict[str, list[Any]]:
    with open(path, encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj["search_space"]


def _normalize_constraints(raw: Any) -> dict[str, bool]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): bool(v) for k, v in raw.items()}
    if isinstance(raw, list):
        merged: dict[str, bool] = {}
        for item in raw:
            if isinstance(item, dict):
                merged.update({str(k): bool(v) for k, v in item.items()})
        return merged
    raise ValueError(f"Unsupported constraints format: {type(raw)}")


def _valid_trial(params: dict, constraints: dict) -> bool:
    if constraints.get("hidden_dim_equals_model_dim"):
        if params["hidden_dim"] != params["model_dim"]:
            return False
    if constraints.get("model_dim_divisible_by_num_attention_heads"):
        if params["model_dim"] % params["num_attention_heads"] != 0:
            return False
    return True


def sample_params(
    search_space: dict[str, list[Any]],
    constraints: dict,
    rng: random.Random,
    max_attempts: int = 200,
) -> dict[str, Any]:
    keys = list(search_space.keys())
    for _ in range(max_attempts):
        params = {key: rng.choice(search_space[key]) for key in keys}
        if constraints.get("hidden_dim_equals_model_dim"):
            params["hidden_dim"] = params["model_dim"]
        if _valid_trial(params, constraints):
            return params
    raise RuntimeError("Could not sample a valid trial; relax constraints or search space.")


def _trial_path(tuning_dir: Path, trial_id: int) -> Path:
    return tuning_dir / f"trial_{trial_id:03d}.json"


def _load_completed_trials(tuning_dir: Path) -> list[dict]:
    trials = []
    for path in sorted(tuning_dir.glob("trial_*.json")):
        with open(path, encoding="utf-8") as f:
            trials.append(json.load(f))
    return trials


def _save_summary(tuning_dir: Path, trials: list[dict]) -> dict:
    ranked = sorted(trials, key=lambda t: t["valid_rocauc"], reverse=True)
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "num_trials": len(ranked),
        "best": ranked[0] if ranked else None,
        "trials": [
            {
                "run_name": t["run_name"],
                "valid_rocauc": t["valid_rocauc"],
                "test_rocauc": t["test_rocauc"],
                "params": {
                    k: t["config"][k]
                    for k in (
                        "lr",
                        "weight_decay",
                        "dropout",
                        "hidden_dim",
                        "model_dim",
                        "num_han_layers",
                        "num_pdg_layers",
                        "num_attention_heads",
                        "batch_size",
                        "pooling",
                    )
                },
            }
            for t in ranked
        ],
    }
    out = tuning_dir / "tuning_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Tune HAN+PDGNN fusion hyperparameters")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--search-space", type=str, default="tune_search.yaml")
    parser.add_argument("--model", type=str, default="main")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-seed", type=int, default=42, help="RNG seed for search sampling")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per trial")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    parser.add_argument("--max-samples", type=int, default=None, help="Subset for quick smoke tuning")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tuning-dir", type=str, default="results/tuning")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save .pt for every trial")
    parser.add_argument("--start-trial", type=int, default=1)
    args = parser.parse_args()

    base_cfg = load_config(PROJECT_ROOT / args.config)
    search_obj = yaml.safe_load((PROJECT_ROOT / args.search_space).read_text(encoding="utf-8"))
    search_space = search_obj["search_space"]
    constraints = _normalize_constraints(search_obj.get("constraints", {}))

    if args.epochs is not None:
        base_cfg["epochs"] = args.epochs
    if args.patience is not None:
        base_cfg["patience"] = args.patience
    if args.device is not None:
        base_cfg["device"] = args.device

    tuning_dir = PROJECT_ROOT / args.tuning_dir
    tuning_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.trial_seed)
    completed = _load_completed_trials(tuning_dir)
    done_ids = {int(t["run_name"].split("_")[-1]) for t in completed if "run_name" in t}

    for trial_idx in range(args.start_trial, args.start_trial + args.n_trials):
        if trial_idx in done_ids:
            print(f"Skip trial_{trial_idx:03d} (already exists)")
            continue

        params = sample_params(search_space, constraints, rng)
        trial_cfg = _deep_merge(base_cfg, params)
        run_name = f"trial_{trial_idx:03d}"

        print(f"\n=== {run_name} ===")
        print(json.dumps(params, indent=2))

        result = run_training(
            trial_cfg,
            args.model,
            seed=args.seed + trial_idx,
            max_samples=args.max_samples,
            run_name=run_name,
            results_subdir="tuning",
            save_checkpoint=args.save_checkpoints,
        )

        with open(_trial_path(tuning_dir, trial_idx), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "sampled_params": params,
                    **result,
                },
                f,
                indent=2,
            )

        completed = _load_completed_trials(tuning_dir)
        summary = _save_summary(tuning_dir, completed)
        if summary["best"]:
            best = summary["best"]
            print(
                f"Best so far: {best['run_name']} | "
                f"valid={best['valid_rocauc']:.4f} | test={best['test_rocauc']:.4f}"
            )

    summary = _save_summary(tuning_dir, _load_completed_trials(tuning_dir))
    print(f"\nTuning complete. Summary: {tuning_dir / 'tuning_summary.json'}")
    if summary.get("best"):
        print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":
    main()
