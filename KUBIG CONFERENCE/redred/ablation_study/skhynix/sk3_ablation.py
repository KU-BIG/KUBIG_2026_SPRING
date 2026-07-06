"""
sk3_ablation.py — Step 3: Ablation Study (v2)
==============================================
기존 지표(RMSE, DirAcc) + 보강 지표(IC, ICIR, Bull/Bear DirAcc) 추가.
Hold-out 평가(2022-01 이후) 분리 추가.

Model A (Full):        wsts_pred_t6 + 거시경제 피처 + SK 자체 피처
Model B (No supply):   거시경제 피처 + SK 자체 피처
Model C (Supply only): [wsts_pred_t6]

IC/ICIR 계산 방식:
  단일 샘플 fold 구조상 fold별 IC 계산 불가 (spearmanr 최소 2 샘플 필요).
  IC    : 전체 CV 구간 예측값 vs 실제값 Spearman rank correlation.
  ICIR  : rolling 12-month window IC의 mean / std 비율.

출력:
  skhynix/outputs/metrics/sk_ablation_results.csv    (기존, 덮어쓰기 금지)
  skhynix/outputs/metrics/sk_ablation_results_v2.csv (신규 CV 결과)
  skhynix/outputs/metrics/sk_holdout_results.csv     (신규 hold-out 결과)
  skhynix/outputs/figures/sk_ablation_comparison.png
  skhynix/outputs/figures/sk_shap_summary.png
  skhynix/outputs/figures/sk_shap_wsts.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from scipy.stats import spearmanr
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "-q"], check=True)
    from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from sk_config import (
    STAGE2_PATH, SUPPLY_FINAL_PKL,
    SK_DATA_DIR, SK_FIG_DIR, SK_METRIC_DIR,
    MIN_TRAIN_M, RANDOM_STATE,
    TARGET_COL, TARGET_NAME,
)

HOLD_OUT_START = "2023-01-01"
IC_WINDOW = 12
SK_PRED_PATH = os.path.join(SK_DATA_DIR, "sk_cv_predictions.parquet")


# ── XGBoost 파라미터 로드 ──────────────────────────────────────
def load_xgb_params() -> dict:
    try:
        with open(SUPPLY_FINAL_PKL, "rb") as f:
            saved = pickle.load(f)
        p = {**saved["best_params"], "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
        print(f"  Stage 1 best_params 로드 (n_estimators={p.get('n_estimators')})")
        return p
    except Exception as e:
        print(f"  [경고] pkl 로드 실패({e}), 기본값 사용")
        return {
            "n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 3,
            "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1,
        }


# ── IC / ICIR 계산 ────────────────────────────────────────────
def compute_ic_icir(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    IC   : 전체 구간 Spearman rank correlation (y_pred, y_true).
    ICIR : rolling IC_WINDOW-month window IC 의 mean / std.
    단일 샘플 fold 구조상 fold별 ICIR 계산 불가 — rolling window 방식 사용.
    """
    if len(y_true) < 2:
        return float("nan"), float("nan")
    ic_overall = float(spearmanr(y_pred, y_true)[0])
    rolling_ics = []
    for i in range(len(y_true) - IC_WINDOW + 1):
        w_ic = float(spearmanr(y_pred[i:i + IC_WINDOW], y_true[i:i + IC_WINDOW])[0])
        if not np.isnan(w_ic):
            rolling_ics.append(w_ic)
    if len(rolling_ics) >= 2:
        icir = float(np.mean(rolling_ics)) / (float(np.std(rolling_ics)) + 1e-9)
    else:
        icir = float("nan")
    return ic_overall, icir


# ── Bull / Bear 분리 방향 정확도 ────────────────────────────────
def compute_bull_bear(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Bull (y_true > 0) : 해당 구간에서 예측이 양(+)으로 맞춘 비율.
    Bear (y_true <= 0): 해당 구간에서 예측이 음(-)으로 맞춘 비율.
    """
    bull_mask = y_true > 0
    bear_mask = ~bull_mask
    n_bull = int(bull_mask.sum())
    n_bear = int(bear_mask.sum())
    bull_acc = float(np.mean(y_pred[bull_mask] > 0) * 100) if n_bull > 0 else float("nan")
    bear_acc = float(np.mean(y_pred[bear_mask] <= 0) * 100) if n_bear > 0 else float("nan")
    return {"bull_acc": bull_acc, "n_bull": n_bull, "bear_acc": bear_acc, "n_bear": n_bear}


# ── 기본 평가 지표 (fold당 1샘플 기준) ──────────────────────────
def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    dir_acc = float(np.mean((y_true > 0) == (y_pred > 0)) * 100)
    return {"rmse": rmse, "mae": mae, "dir_acc": dir_acc}


# ── Walk-Forward CV ───────────────────────────────────────────
def walk_forward_cv(X: pd.DataFrame, y: pd.Series, params: dict):
    """
    Returns:
        fold_results : list[dict]  — fold별 기본 지표 (rmse, mae, dir_acc)
        all_preds    : np.ndarray  — 전체 fold 예측값 시퀀스
        all_trues    : np.ndarray  — 전체 fold 실제값 시퀀스
    """
    n = len(X)
    fold_results, all_preds, all_trues = [], [], []
    for test_end in range(MIN_TRAIN_M, n):
        X_tr = X.iloc[:test_end].values
        y_tr = y.iloc[:test_end].values
        X_te = X.iloc[test_end:test_end + 1].values
        y_te = y.iloc[test_end:test_end + 1].values
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        fold_results.append(compute_fold_metrics(y_te, pred))
        all_preds.append(float(pred[0]))
        all_trues.append(float(y_te[0]))
    return fold_results, np.array(all_preds), np.array(all_trues)


def summarize_cv(fold_results: list, all_preds: np.ndarray, all_trues: np.ndarray) -> dict:
    """fold 결과 집계 + IC / ICIR / Bull/Bear 추가."""
    summary = {}
    for k in ["rmse", "mae", "dir_acc"]:
        vals = [r[k] for r in fold_results if not np.isnan(r[k])]
        summary[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        summary[f"{k}_std"]  = float(np.std(vals))  if vals else float("nan")
    ic, icir = compute_ic_icir(all_trues, all_preds)
    summary["ic"]   = ic
    summary["icir"] = icir
    summary.update(compute_bull_bear(all_trues, all_preds))
    return summary


# ── Hold-out 평가 ─────────────────────────────────────────────
def evaluate_holdout(
    cv_X: pd.DataFrame, cv_y: pd.Series,
    ho_X: pd.DataFrame, ho_y: pd.Series,
    params: dict,
) -> dict:
    """CV 전체 데이터로 학습 후 hold-out 구간 예측, 전체 지표 반환."""
    m = xgb.XGBRegressor(**params)
    m.fit(cv_X.values, cv_y.values)
    preds = m.predict(ho_X.values)
    trues = ho_y.values
    rmse    = float(np.sqrt(mean_squared_error(trues, preds)))
    mae     = float(mean_absolute_error(trues, preds))
    dir_acc = float(np.mean((trues > 0) == (preds > 0)) * 100)
    ic, icir = compute_ic_icir(trues, preds)
    return {"rmse": rmse, "mae": mae, "dir_acc": dir_acc,
            "ic": ic, "icir": icir, **compute_bull_bear(trues, preds)}


# ── 포맷 헬퍼 ─────────────────────────────────────────────────
def _f(val, spec=".2f", suffix=""):
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    try:
        return format(val, spec) + suffix
    except Exception:
        return str(val)


# ── 결과 출력 — Walk-forward CV ───────────────────────────────
def print_cv_table(summaries: dict):
    W, CW = 64, 14
    print(f"\n{'═'*W}")
    print(f"  Walk-forward CV Results — [{TARGET_NAME}]")
    print(f"  ※ CV 구간 only (< {HOLD_OUT_START}), 기존 전체 구간 결과와 수치 다를 수 있음")
    print(f"{'═'*W}")
    print(f"  {'지표':<20} {'Model A':>{CW}} {'Model B':>{CW}} {'Model C':>{CW}}")
    print(f"  {'─'*62}")

    def row_ms(label, key, spec=".2f", suffix=""):
        cells = ""
        for mk in ["A", "B", "C"]:
            s = summaries.get(mk, {})
            m  = s.get(f"{key}_mean", float("nan"))
            sd = s.get(f"{key}_std",  float("nan"))
            cell = f"{_f(m, spec, suffix)}±{_f(sd, spec, suffix)}" if not np.isnan(m) else "N/A"
            cells += f" {cell:>{CW}}"
        print(f"  {label:<20}{cells}")

    def row_val(label, key, spec=".2f", suffix=""):
        cells = ""
        for mk in ["A", "B", "C"]:
            v = summaries.get(mk, {}).get(key, float("nan"))
            cells += f" {_f(v, spec, suffix):>{CW}}"
        print(f"  {label:<20}{cells}")

    row_ms("RMSE",    "rmse")
    row_ms("Dir Acc", "dir_acc", suffix="%")

    print(f"  {'─'*62}")
    n_bull = summaries.get("A", {}).get("n_bull", "?")
    n_bear = summaries.get("A", {}).get("n_bear", "?")
    row_val(f"Bull Acc (n={n_bull})", "bull_acc", ".1f", "%")
    row_val(f"Bear Acc (n={n_bear})", "bear_acc", ".1f", "%")

    print(f"  {'─'*62}")
    row_val("IC",   "ic",   ".4f")
    row_val("ICIR", "icir", ".3f")

    sa, sb = summaries.get("A", {}), summaries.get("B", {})
    d_rmse = sa.get("rmse_mean", float("nan")) - sb.get("rmse_mean", float("nan"))
    d_dir  = sa.get("dir_acc_mean", float("nan")) - sb.get("dir_acc_mean", float("nan"))
    d_ic   = sa.get("ic", float("nan")) - sb.get("ic", float("nan"))
    print(f"\n  Δ A vs B (공급 신호 incremental 기여):")
    print(f"    RMSE    : {_f(d_rmse, '+.2f')}  ({'Model A 우수' if d_rmse < 0 else 'Model B 우수'})")
    print(f"    Dir Acc : {_f(d_dir,  '+.2f')}%p ({'Model A 우수' if d_dir > 0 else 'Model B 우수'})")
    print(f"    IC      : {_f(d_ic,   '+.4f')}")
    if not np.isnan(d_rmse):
        verdict = "공급 신호 유효 (RMSE 또는 DirAcc 기준)" if (d_rmse < 0 or d_dir > 0) else "공급 신호 유효하지 않음"
        print(f"  결론: {verdict}")
    print(f"{'═'*W}")


# ── 결과 출력 — Hold-out ─────────────────────────────────────
def print_holdout_table(summaries_ho: dict, n_holdout: int):
    W, CW = 64, 14
    print(f"\n{'═'*W}")
    print(f"  Hold-out Results — [{TARGET_NAME}]")
    print(f"  ({HOLD_OUT_START} 이후,  n={n_holdout}개월)")
    print(f"{'═'*W}")
    print(f"  {'지표':<20} {'Model A':>{CW}} {'Model B':>{CW}} {'Model C':>{CW}}")
    print(f"  {'─'*62}")

    def row_val(label, key, spec=".2f", suffix=""):
        cells = ""
        for mk in ["A", "B", "C"]:
            v = summaries_ho.get(mk, {}).get(key, float("nan"))
            cells += f" {_f(v, spec, suffix):>{CW}}"
        print(f"  {label:<20}{cells}")

    row_val("RMSE",    "rmse")
    row_val("Dir Acc", "dir_acc", ".1f", "%")

    print(f"  {'─'*62}")
    n_bull = summaries_ho.get("A", {}).get("n_bull", "?")
    n_bear = summaries_ho.get("A", {}).get("n_bear", "?")
    row_val(f"Bull Acc (n={n_bull})", "bull_acc", ".1f", "%")
    row_val(f"Bear Acc (n={n_bear})", "bear_acc", ".1f", "%")

    print(f"  {'─'*62}")
    row_val("IC",   "ic",   ".4f")
    row_val("ICIR", "icir", ".3f")
    print(f"{'═'*W}")


# ── 결과 출력 — CV vs Hold-out (Model A 기준) ─────────────────
def print_cv_vs_holdout(cv_s: dict, ho_s: dict):
    W, CW = 58, 12
    print(f"\n{'═'*W}")
    print(f"  CV vs Hold-out 비교 (Model A 기준)")
    print(f"{'═'*W}")
    print(f"  {'지표':<16} {'CV':>{CW}} {'Hold-out':>{CW}} {'차이':>{CW}}")
    print(f"  {'─'*54}")
    comparisons = [
        ("Dir Acc",  "dir_acc_mean", "dir_acc", ".1f", "%"),
        ("IC",       "ic",           "ic",       ".4f", ""),
        ("Bull Acc", "bull_acc",     "bull_acc", ".1f", "%"),
        ("Bear Acc", "bear_acc",     "bear_acc", ".1f", "%"),
    ]
    for label, cv_key, ho_key, spec, suffix in comparisons:
        cv_v = cv_s.get(cv_key, float("nan"))
        ho_v = ho_s.get(ho_key, float("nan"))
        if not (np.isnan(cv_v) or np.isnan(ho_v)):
            diff = ho_v - cv_v
            sign = "+" if diff >= 0 else ""
            diff_s = sign + _f(diff, spec, suffix)
        else:
            diff_s = "N/A"
        print(f"  {label:<16} {_f(cv_v, spec, suffix):>{CW}} {_f(ho_v, spec, suffix):>{CW}} {diff_s:>{CW}}")
    print(f"{'═'*W}")


# ── SHAP 분석 ─────────────────────────────────────────────────
def run_shap(model, X: pd.DataFrame):
    try:
        import shap
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "shap", "-q"], check=True)
        import shap

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, _ = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title(f"SHAP Summary — {TARGET_NAME}")
    plt.tight_layout()
    path = os.path.join(SK_FIG_DIR, "sk_shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")

    if "wsts_pred_t6" in X.columns:
        idx = list(X.columns).index("wsts_pred_t6")
        wsts_shap = shap_values[:, idx]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(wsts_shap, bins=30, edgecolor="white", color="steelblue")
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
        ax.set_xlabel("SHAP value (wsts_pred_t6)")
        ax.set_ylabel("Count")
        ax.set_title(f"wsts_pred_t6 SHAP Distribution — {TARGET_NAME}")
        plt.tight_layout()
        path2 = os.path.join(SK_FIG_DIR, "sk_shap_wsts.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {path2}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking  = sorted(zip(X.columns, mean_abs), key=lambda x: -x[1])
    print("  평균 |SHAP| 피처 랭킹 (상위 10):")
    for rank, (col, val) in enumerate(ranking[:10], 1):
        marker = " ←" if col == "wsts_pred_t6" else ""
        print(f"    {rank:2d}. {col:<35s} {val:.4f}{marker}")


# ── 예측값 추출 (parquet 저장용) ──────────────────────────────
def _holdout_preds(cv_X: pd.DataFrame, cv_y: pd.Series,
                   ho_X: pd.DataFrame, params: dict) -> np.ndarray:
    """CV 전체로 학습 후 hold-out 예측값만 반환 (지표 계산 없음)."""
    m = xgb.XGBRegressor(**params)
    m.fit(cv_X.values, cv_y.values)
    return m.predict(ho_X.values)


# ── 결과 바차트 ────────────────────────────────────────────────
def plot_comparison(summaries: dict, title_suffix: str = "CV"):
    metrics      = ["rmse", "mae", "dir_acc"]
    metric_labels = {"rmse": "RMSE", "mae": "MAE", "dir_acc": "Dir Acc (%)"}
    model_keys   = ["A", "B", "C"]
    model_labels  = {"A": "Model A\n(Full)", "B": "Model B\n(No supply)", "C": "Model C\n(Supply only)"}
    colors        = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        vals, errs, lbls = [], [], []
        for mk in model_keys:
            s = summaries.get(mk, {})
            vals.append(s.get(f"{metric}_mean", s.get(metric, 0)))
            errs.append(s.get(f"{metric}_std", 0))
            lbls.append(model_labels[mk])
        bars = ax.bar(lbls, vals, yerr=errs, capsize=5,
                      color=[colors[k] for k in model_keys], alpha=0.85)
        ax.set_title(metric_labels[metric])
        ax.set_ylabel(metric_labels[metric])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"Ablation Study: Supply Signal Contribution\n({TARGET_NAME} — {title_suffix})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(SK_FIG_DIR, "sk_ablation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ── 메인 ─────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  Step 3  Ablation Study v2")
    print("  지표 보강: IC / ICIR / Bull/Bear DirAcc / Hold-out")
    print("=" * 64)

    # ── [1] 데이터 로드 및 분리 ────────────────────────────────
    print("\n[1] 데이터 로드 및 분리")
    df = pd.read_parquet(STAGE2_PATH)
    print(f"  전체: {df.shape}  "
          f"날짜: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")

    ho_ts  = pd.Timestamp(HOLD_OUT_START)
    df_cv  = df[df.index < ho_ts]
    df_ho  = df[df.index >= ho_ts]
    print(f"  CV 구간       : {len(df_cv)}행  "
          f"{df_cv.index[0].strftime('%Y-%m')} ~ {df_cv.index[-1].strftime('%Y-%m')}")
    print(f"  Hold-out 구간 : {len(df_ho)}행  "
          f"{df_ho.index[0].strftime('%Y-%m')} ~ {df_ho.index[-1].strftime('%Y-%m')}")
    print(f"  ※ Walk-forward CV는 CV 구간만 사용 — 기존 전체 구간 결과와 수치 상이할 수 있음")

    feature_all = [c for c in df.columns if c != TARGET_COL]
    macro_sk    = [c for c in feature_all if c != "wsts_pred_t6"]
    supply_only = ["wsts_pred_t6"]
    feature_sets = {"A": feature_all, "B": macro_sk, "C": supply_only}
    model_names  = {
        "A": "Model A (Full)",
        "B": "Model B (No supply)",
        "C": "Model C (Supply only)",
    }
    print(f"\n  피처셋: A={len(feature_all)}개, B={len(macro_sk)}개, C={len(supply_only)}개")

    # ── [2] XGBoost 파라미터 ────────────────────────────────────
    print("\n[2] XGBoost 파라미터 로드")
    params = load_xgb_params()

    y_cv = df_cv[TARGET_COL]
    y_ho = df_ho[TARGET_COL]

    # ── [3] Walk-Forward CV ─────────────────────────────────────
    print(f"\n[3] Walk-forward CV  (CV 구간,  min_train={MIN_TRAIN_M})")
    summaries_cv        = {}
    cv_rows             = []
    cv_preds_by_model: dict = {}
    cv_trues_ref        = None

    for mk, feat_cols in feature_sets.items():
        print(f"\n  [{model_names[mk]}] 실행 중...")
        X_cv = df_cv[feat_cols]
        fold_results, all_preds, all_trues = walk_forward_cv(X_cv, y_cv, params)
        summary = summarize_cv(fold_results, all_preds, all_trues)
        summaries_cv[mk] = summary
        cv_preds_by_model[mk] = all_preds
        if mk == "A":
            cv_trues_ref = all_trues
        print(f"    n_folds={len(fold_results)},  "
              f"RMSE={summary['rmse_mean']:.2f}±{summary['rmse_std']:.2f},  "
              f"DirAcc={summary['dir_acc_mean']:.1f}%,  "
              f"IC={_f(summary['ic'], '.4f')},  "
              f"ICIR={_f(summary['icir'], '.3f')}")
        print(f"    BullAcc={_f(summary['bull_acc'], '.1f')}%(n={summary['n_bull']}),  "
              f"BearAcc={_f(summary['bear_acc'], '.1f')}%(n={summary['n_bear']})")
        cv_rows.append({
            "model":    mk,
            "rmse":     summary["rmse_mean"],
            "mae":      summary["mae_mean"],
            "dir_acc":  summary["dir_acc_mean"],
            "bull_acc": summary["bull_acc"],
            "n_bull":   summary["n_bull"],
            "bear_acc": summary["bear_acc"],
            "n_bear":   summary["n_bear"],
            "ic":       summary["ic"],
            "icir":     summary["icir"],
        })

    print_cv_table(summaries_cv)

    # ── [4] Hold-out 평가 ───────────────────────────────────────
    print(f"\n[4] Hold-out 평가  ({HOLD_OUT_START} 이후,  n={len(df_ho)})")
    summaries_ho        = {}
    ho_rows             = []
    ho_preds_by_model: dict = {}

    for mk, feat_cols in feature_sets.items():
        print(f"\n  [{model_names[mk]}] 실행 중...")
        X_cv_m = df_cv[feat_cols]
        X_ho_m = df_ho[feat_cols]
        ho_m   = evaluate_holdout(X_cv_m, y_cv, X_ho_m, y_ho, params)
        summaries_ho[mk] = ho_m
        ho_preds_by_model[mk] = _holdout_preds(X_cv_m, y_cv, X_ho_m, params)
        print(f"    RMSE={ho_m['rmse']:.2f},  "
              f"DirAcc={ho_m['dir_acc']:.1f}%,  "
              f"IC={_f(ho_m['ic'], '.4f')},  "
              f"ICIR={_f(ho_m['icir'], '.3f')}")
        print(f"    BullAcc={_f(ho_m['bull_acc'], '.1f')}%(n={ho_m['n_bull']}),  "
              f"BearAcc={_f(ho_m['bear_acc'], '.1f')}%(n={ho_m['n_bear']})")
        ho_rows.append({"model": mk, **ho_m})

    print_holdout_table(summaries_ho, len(df_ho))
    print_cv_vs_holdout(summaries_cv["A"], summaries_ho["A"])

    # ── [5] SHAP (CV 구간 전체로 학습한 Model A) ──────────────
    print(f"\n[5] SHAP 분석  (CV 전체 데이터, Model A)")
    X_cv_all    = df_cv[feature_sets["A"]]
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X_cv_all.values, y_cv.values)
    run_shap(final_model, X_cv_all)

    # ── [6] 결과 저장 ───────────────────────────────────────────
    print("\n[6] 결과 저장")
    csv_v2 = os.path.join(SK_METRIC_DIR, "sk_ablation_results_v2.csv")
    csv_ho = os.path.join(SK_METRIC_DIR, "sk_holdout_results.csv")
    pd.DataFrame(cv_rows).to_csv(csv_v2, index=False)
    pd.DataFrame(ho_rows).to_csv(csv_ho, index=False)
    print(f"  → {csv_v2}")
    print(f"  → {csv_ho}")
    print(f"  (sk_ablation_results.csv 는 그대로 유지됨)")

    # ── 예측값 parquet 저장 ──────────────────────────────────
    cv_dates = df_cv.index[MIN_TRAIN_M:]
    pred_cv = pd.DataFrame({
        "date":     cv_dates,
        "y_true":   cv_trues_ref,
        "y_pred_A": cv_preds_by_model["A"],
        "y_pred_B": cv_preds_by_model["B"],
        "y_pred_C": cv_preds_by_model["C"],
        "split":    "cv",
    })
    pred_ho = pd.DataFrame({
        "date":     df_ho.index,
        "y_true":   y_ho.values,
        "y_pred_A": ho_preds_by_model["A"],
        "y_pred_B": ho_preds_by_model["B"],
        "y_pred_C": ho_preds_by_model["C"],
        "split":    "holdout",
    })
    pred_all = pd.concat([pred_cv, pred_ho], ignore_index=True)
    pred_all.to_parquet(SK_PRED_PATH, index=False)
    print(f"  → {SK_PRED_PATH}  ({len(pred_all)}행)")

    # ── [7] 바차트 (CV 기준) ────────────────────────────────────
    # plot_comparison expects dict with {metric}_mean keys; summaries_cv has them
    plot_comparison(summaries_cv, title_suffix="Walk-forward CV")
    print("\n  Step 3 v2 완료.")


if __name__ == "__main__":
    main()
