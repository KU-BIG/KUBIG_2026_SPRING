#!/usr/bin/env python3
"""Class-wise mixing topology ablation 결과를 한국어 Markdown으로 요약한다.

runs/class_wise_mixing/*/metrics.json만 읽고, smoke-test seed 0은 제외한다.
기본 출력은 results/CLASS_WISE_MIXING.md이다.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

DATASETS = ["acm", "dblp", "imdb", "freebase", "mag", "aifb", "yelp"]
BACKBONES = ["han", "rgcn"]
SEEDS = [312132, 238623, 792965, 15092, 661491, 588722, 825661, 500973, 88015, 251219]
ROOT = Path("runs/class_wise_mixing")
OUT = Path("results/CLASS_WISE_MIXING.md")


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def has_nan(x: Any) -> bool:
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, dict):
        return any(has_nan(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(has_nan(v) for v in x)
    return False


def fmt_num(x: Any, digits: int = 4) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if not is_finite_number(x):
        return "-"
    return f"{float(x):.{digits}f}"


def fmt_mean_std(vals: list[Any], digits: int = 4) -> str:
    xs = [float(v) for v in vals if is_finite_number(v)]
    if not xs:
        return "-"
    sd = stdev(xs) if len(xs) > 1 else 0.0
    return f"{mean(xs):.{digits}f}+/-{sd:.{digits}f}"


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ds in DATASETS:
        for bb in BACKBONES:
            for seed in SEEDS:
                path = ROOT / f"{ds}__{bb}__class_wise_mixing_s{seed}" / "metrics.json"
                if not path.exists():
                    missing.append({"dataset": ds, "backbone": bb, "seed": seed, "path": str(path)})
                    continue
                m = json.loads(path.read_text())
                mix = m.get("class_wise_mixing") or {}
                mixed = mix.get("mixed_nodes")
                unchanged = mix.get("unchanged_nodes")
                denom = (mixed or 0) + (unchanged or 0) if mixed is not None and unchanged is not None else 0
                ratio = float(mixed) / denom if denom else None
                final_finite = all(is_finite_number(m.get(k)) for k in ("test_macro_f1", "test_accuracy", "val_macro_f1"))
                rows.append({
                    "dataset": ds,
                    "backbone": bb,
                    "seed": seed,
                    "path": str(path),
                    "test_macro_f1": m.get("test_macro_f1"),
                    "test_accuracy": m.get("test_accuracy"),
                    "val_macro_f1": m.get("val_macro_f1"),
                    "gtn_only_test_macro_f1": m.get("gtn_only_test_macro_f1"),
                    "gtn_only_test_accuracy": m.get("gtn_only_test_accuracy"),
                    "mixed_nodes": mixed,
                    "unchanged_nodes": unchanged,
                    "mixed_ratio": ratio,
                    "nan_gtn_attention": has_nan(m.get("gtn_attentions")),
                    "final_metrics_finite": final_finite,
                })
    return rows, missing


def group(rows: list[dict[str, Any]], ds: str, bb: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["dataset"] == ds and r["backbone"] == bb]


def summary_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 데이터셋 | 백본 | n | 유한 지표 n | test_macro_f1 평균+/-표본표준편차 | test_accuracy 평균+/-표본표준편차 | val_macro_f1 평균+/-표본표준편차 | mixed ratio 평균+/-표본표준편차 | GTN attention NaN run 수 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for bb in BACKBONES:
            xs = group(rows, ds, bb)
            if not xs:
                continue
            finite = [r for r in xs if r["final_metrics_finite"]]
            ratios = [r["mixed_ratio"] for r in xs if is_finite_number(r.get("mixed_ratio"))]
            lines.append("| " + " | ".join([
                ds,
                bb.upper(),
                str(len(xs)),
                str(len(finite)),
                fmt_mean_std([r["test_macro_f1"] for r in finite]),
                fmt_mean_std([r["test_accuracy"] for r in finite]),
                fmt_mean_std([r["val_macro_f1"] for r in finite]),
                fmt_mean_std(ratios),
                str(sum(1 for r in xs if r["nan_gtn_attention"])),
            ]) + " |")
    return lines


def per_run_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 데이터셋 | 백본 | seed | test_macro_f1 | test_accuracy | val_macro_f1 | mixed_nodes | unchanged_nodes | mixed_ratio | gtn_only_test_macro_f1 | gtn_only_test_accuracy | GTN attention NaN | 최종 지표 유한 | metrics 경로 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            r["dataset"],
            r["backbone"].upper(),
            str(r["seed"]),
            fmt_num(r["test_macro_f1"]),
            fmt_num(r["test_accuracy"]),
            fmt_num(r["val_macro_f1"]),
            str(r["mixed_nodes"] if r["mixed_nodes"] is not None else "-"),
            str(r["unchanged_nodes"] if r["unchanged_nodes"] is not None else "-"),
            fmt_num(r["mixed_ratio"]),
            fmt_num(r["gtn_only_test_macro_f1"]),
            fmt_num(r["gtn_only_test_accuracy"]),
            "예" if r["nan_gtn_attention"] else "아니오",
            "예" if r["final_metrics_finite"] else "아니오",
            f"`{r['path']}`",
        ]) + " |")
    return lines


def missing_section(missing: list[dict[str, Any]]) -> list[str]:
    if not missing:
        return ["없음."]
    return [f"- {m['dataset']} / {m['backbone'].upper()} seed={m['seed']}: `{m['path']}`" for m in missing]


def diagnostics(rows: list[dict[str, Any]]) -> list[str]:
    nan_groups = []
    for ds in DATASETS:
        for bb in BACKBONES:
            xs = group(rows, ds, bb)
            n_nan = sum(1 for r in xs if r["nan_gtn_attention"])
            if n_nan:
                nan_groups.append(f"{ds}/{bb.upper()} {n_nan}/{len(xs)}")
    bad_final = [r for r in rows if not r["final_metrics_finite"]]
    freebase = [r for r in rows if r["dataset"] == "freebase"]
    fb_ratios = [r["mixed_ratio"] for r in freebase if is_finite_number(r.get("mixed_ratio"))]
    return [
        "- GTN attention NaN이 있는 그룹: " + ("; ".join(nan_groups) if nan_groups else "없음") + ".",
        "- 최종 지표 NaN 예외: " + (f"{len(bad_final)}/{len(rows)} run: " + ", ".join(f"`{r['path']}`" for r in bad_final) if bad_final else "없음") + ".",
        f"- Freebase GTN attention caveat: Freebase 20개 run 중 {sum(1 for r in freebase if r['nan_gtn_attention'])}/20개에서 `gtn_attentions`에 NaN이 있다. 최종 지표는 {sum(1 for r in freebase if r['final_metrics_finite'])}/20개 run에서 유한하다.",
        f"- Freebase mixed ratio: {fmt_mean_std(fb_ratios)}.",
        "- IMDB도 저장된 `gtn_attentions`에는 NaN이 있지만 최종 지표는 유한하다. 이 값은 숨기지 않고 attention artifact 진단 caveat로 보고한다.",
        "- 같은 class 및 같은 split 안의 group이 작으면 class-wise mixing이 약해질 수 있다. singleton group은 그대로 남으며 `unchanged_nodes`와 mixed ratio에 반영된다.",
        "- MAG는 class 수가 많고 subsampled node를 사용하므로 macro-F1 절대값이 낮을 수 있다. Yelp는 HNE featureless multilabel 설정이라 macro-F1을 accuracy 및 같은 dataset 안의 조건 간 차이와 함께 해석해야 한다.",
    ]


def build_markdown(rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    md: list[str] = [
        "# 클래스별 혼합 위상 Ablation",
        "",
        "## 상태",
        "",
        f"- 생성 시각: {generated}",
        f"- 완료 상태: `completed={len(rows)} missing={len(missing)} expected=140`",
        "- 공식 run: 7 datasets x 2 backbones x 10 seeds = 140 runs",
        f"- 최종 지표가 모두 유한한 run: {sum(1 for r in rows if r['final_metrics_finite'])}/{len(rows)}",
        "- smoke-test seed `0`은 모든 요약에서 제외했다.",
        "- 공식 seed: " + ", ".join(map(str, SEEDS)) + ".",
        "",
        "## 실험 설정",
        "",
        "- Dataset: `acm`, `dblp`, `imdb`, `freebase`, `mag`, `aifb`, `yelp`.",
        "- Backbone: HAN, RGCN.",
        "- Topology source: `gtn`.",
        "- Topology mode: `class_wise_mixing`.",
        "- 결과 경로 패턴: `runs/class_wise_mixing/<dataset>__<backbone>__class_wise_mixing_s<seed>/metrics.json`.",
        "- Topology cache는 class-wise mixing 이전의 실제 GTN-PDGNN 위상 특징만 저장한다. 이는 계산 재사용일 뿐이며 모델 또는 ablation 정의를 바꾸지 않는다.",
        "",
        "## 동기와 해석",
        "",
        "Class-wise mixing은 실제 GTN-PDGNN 위상 특징을 계산한 뒤, backbone feature와 concat하기 전에 각 노드의 위상 특징을 같은 class, 가능하면 같은 train/val/test split 안의 다른 노드 위상 특징으로 교체하는 구조적 대조군이다. class-level 위상 분포는 일부 보존하면서, 특정 노드와 특정 위상 특징 사이의 정렬만 깨뜨린다.",
        "",
        "Random/noisy feature 대조군은 단순 차원 추가 효과를 검정한다. Class-wise mixing은 그보다 강한 반사실적 대조군으로, real GTN-PDGNN topology가 class-wise mixed topology보다 좋을 때 node-specific topology alignment가 의미 있다는 해석을 뒷받침한다.",
        "",
        "## 현재 결과 요약표",
        "",
        *summary_table(rows),
        "",
        "## 미완료 공식 Run",
        "",
        *missing_section(missing),
        "",
        "## Run별 상세표",
        "",
        *per_run_table(rows),
        "",
        "## 진단 / Caveat",
        "",
        *diagnostics(rows),
        "",
        "## 재현 명령",
        "",
        "완료 run 수 확인:",
        "",
        "```bash",
        "find runs/class_wise_mixing -name metrics.json | grep -Ev '_s0/metrics.json$' | wc -l",
        "```",
        "",
        "누락 run 확인:",
        "",
        "```bash",
        "python - <<'PY'",
        "from pathlib import Path",
        "datasets=['acm','dblp','imdb','freebase','mag','aifb','yelp']",
        "backbones=['han','rgcn']",
        "seeds=[312132,238623,792965,15092,661491,588722,825661,500973,88015,251219]",
        "root=Path('runs/class_wise_mixing')",
        "missing=[]",
        "for ds in datasets:",
        "  for bb in backbones:",
        "    for s in seeds:",
        "      p=root/f'{ds}__{bb}__class_wise_mixing_s{s}'/'metrics.json'",
        "      if not p.exists(): missing.append(str(p))",
        "print(f'completed={140-len(missing)} missing={len(missing)} expected=140')",
        "print('\\n'.join(missing))",
        "PY",
        "```",
        "",
        "요약 재생성:",
        "",
        "```bash",
        "python experiments/summarize_class_wise_mixing.py",
        "```",
        "",
        "재실행/이어 실행:",
        "",
        "```bash",
        "source /opt/miniforge3/etc/profile.d/conda.sh",
        "conda activate tda",
        "cd /workspace/TDA",
        "bash experiments/run_class_wise_mixing.sh",
        "```",
        "",
        "이미 `metrics.json`이 있는 run은 `experiments/run_class_wise_mixing.sh`에서 skip된다.",
        "",
    ]
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    rows, missing = load_rows()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_markdown(rows, missing))
    print("\n".join(summary_table(rows)))
    print(f"\nwrote {out} ({len(rows)}/140 official runs, missing {len(missing)})")


if __name__ == "__main__":
    main()
