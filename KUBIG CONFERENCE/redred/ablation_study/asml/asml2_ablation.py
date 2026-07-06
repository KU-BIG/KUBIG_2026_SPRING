"""
asml2_ablation.py — Step 2: ASML Ablation Study + SHAP

Model A (Full):        wsts_pred_t6 + macro + SOX + FX + PCE + semicap + semicapu + ASML self
Model B (No supply):   macro + SOX + FX + PCE + semicap + semicapu + ASML self
Model C (Supply only): wsts_pred_t6만

XGBoost: model/outputs/models/best_xgboost_final.pkl best_params 재사용
CV:      expanding walk-forward, min_train=60, step=1
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asml_config import (
    ASML_FEATURES_PATH, SUPPLY_FINAL_PKL,
    ASML_FIG_DIR, ASML_METRIC_DIR,
    TARGET_COL, SUPPLY_COL, MIN_TRAIN_M, RANDOM_STATE,
)

# sk5 고정값 (sk5_crossfirm.py ASML 결과)
SK5_RMSE_A   = 12.8970
SK5_RMSE_B   = 12.9529
SK5_DIRACC_A = 79.8780
SK5_DIRACC_B = 78.0488


def load_xgb_params() -> dict:
    try:
        with open(SUPPLY_FINAL_PKL, "rb") as f:
            saved = pickle.load(f)
        p = {**saved["best_params"], "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
        print(f"Stage 1 best_params 로드 (n_estimators={p['n_estimators']}, "
              f"lr={p['learning_rate']:.4f}, max_depth={p['max_depth']})")
        return p
    except Exception as e:
        print(f"[경고] pkl 로드 실패({e}), 기본값 사용")
        return {
            "n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 3,
            "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1,
        }


def walk_forward_cv(X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    rmse_list, diracc_list = [], []
    for test_end in range(MIN_TRAIN_M, len(X)):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[:test_end].values, y.iloc[:test_end].values)
        pred = m.predict(X.iloc[test_end : test_end + 1].values)
        y_te = y.iloc[test_end : test_end + 1].values
        rmse_list.append(float(np.sqrt(mean_squared_error(y_te, pred))))
        diracc_list.append(float(((y_te > 0) == (pred > 0)).mean() * 100))
    return {
        "rmse_mean":   float(np.mean(rmse_list)),
        "rmse_std":    float(np.std(rmse_list)),
        "diracc_mean": float(np.mean(diracc_list)),
        "diracc_std":  float(np.std(diracc_list)),
    }


def run_shap(model: xgb.XGBRegressor, X: pd.DataFrame, label: str) -> None:
    try:
        import shap
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "shap", "-q"], check=True)
        import shap

    print("  SHAP TreeExplainer 실행 중...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    fig, _ = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title(f"SHAP Summary — {label} 6-Month Forward Return")
    plt.tight_layout()
    path_summary = os.path.join(ASML_FIG_DIR, "asml_shap_summary.png")
    plt.savefig(path_summary, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path_summary}")

    # wsts_pred_t6 SHAP 분포
    if SUPPLY_COL in X.columns:
        idx       = list(X.columns).index(SUPPLY_COL)
        wsts_shap = shap_values[:, idx]
        fig, ax   = plt.subplots(figsize=(7, 4))
        ax.hist(wsts_shap, bins=30, edgecolor="white", color="steelblue")
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
        ax.set_xlabel(f"SHAP value ({SUPPLY_COL})")
        ax.set_ylabel("Count")
        ax.set_title(f"wsts_pred_t6 SHAP Distribution — {label}")
        plt.tight_layout()
        path_wsts = os.path.join(ASML_FIG_DIR, "asml_shap_wsts.png")
        plt.savefig(path_wsts, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  저장: {path_wsts}")

    # 평균 |SHAP| 랭킹
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking  = sorted(zip(X.columns, mean_abs), key=lambda x: -x[1])
    print("  평균 |SHAP| 피처 랭킹 (상위 10):")
    for rank, (col, val) in enumerate(ranking[:10], 1):
        marker = " ←" if col == SUPPLY_COL else ""
        print(f"    {rank:2d}. {col:<35s} {val:.4f}{marker}")


def print_comparison(res_A: dict, res_B: dict) -> None:
    asl_delta_rmse   = res_A["rmse_mean"]   - res_B["rmse_mean"]
    asl_delta_diracc = res_A["diracc_mean"] - res_B["diracc_mean"]
    sk5_delta_rmse   = SK5_RMSE_A   - SK5_RMSE_B
    sk5_delta_diracc = SK5_DIRACC_A - SK5_DIRACC_B

    w = "═" * 66
    print(f"\n{w}")
    print("  ASML Ablation: sk5 (generic) vs asml2 (ASML-optimized)")
    print(w)
    print(f"  {'지표':<14}{'sk5 A':>12}{'sk5 B':>12}{'asml A':>12}{'asml B':>12}")
    print("  " + "─" * 62)
    print(f"  {'RMSE':<14}{SK5_RMSE_A:>12.2f}{SK5_RMSE_B:>12.2f}"
          f"{res_A['rmse_mean']:>12.2f}{res_B['rmse_mean']:>12.2f}")
    print(f"  {'DirAcc':<14}{SK5_DIRACC_A:>11.1f}%{SK5_DIRACC_B:>11.1f}%"
          f"{res_A['diracc_mean']:>11.1f}%{res_B['diracc_mean']:>11.1f}%")
    print("  " + "─" * 62)
    print(f"  {'Δ RMSE (A-B)':<14}{sk5_delta_rmse:>+12.2f}{'':>12}"
          f"{asl_delta_rmse:>+12.2f}{'':>12}")
    print(f"  {'Δ DirAcc':<14}{sk5_delta_diracc:>+11.1f}%p{'':>12}"
          f"{asl_delta_diracc:>+11.1f}%p{'':>12}")
    print(w)

    improved = []
    if asl_delta_rmse < sk5_delta_rmse:
        improved.append(f"RMSE 기여 개선 (sk5:{sk5_delta_rmse:+.2f} → asml2:{asl_delta_rmse:+.2f})")
    if asl_delta_diracc > sk5_delta_diracc:
        improved.append(f"DirAcc 기여 개선 (sk5:{sk5_delta_diracc:+.1f}%p → asml2:{asl_delta_diracc:+.1f}%p)")
    if improved:
        print(f"  결론: 피처셋 최적화 효과 확인 — {', '.join(improved)}")
    else:
        print("  결론: 피처셋 최적화만으로 혼재 신호 해소 불충분")
    print()


def main() -> None:
    print("=" * 55)
    print("  asml2_ablation.py — ASML Ablation Study + SHAP")
    print("=" * 55)

    df     = pd.read_parquet(ASML_FEATURES_PATH)
    params = load_xgb_params()

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]
    n_folds = len(df) - MIN_TRAIN_M

    feats_A = feature_cols
    feats_B = [c for c in feature_cols if c != SUPPLY_COL]
    feats_C = [SUPPLY_COL]

    print(f"\n데이터: {df.shape}, {n_folds} folds")
    print(f"날짜 범위: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
    print(f"피처: A={len(feats_A)}, B={len(feats_B)}, C={len(feats_C)}")
    print(f"피처 목록 (A): {feats_A}")

    print("\nModel A (Full) 실행 중...")
    res_A = walk_forward_cv(X[feats_A], y, params)
    print(f"  RMSE={res_A['rmse_mean']:.2f}±{res_A['rmse_std']:.2f}, "
          f"DirAcc={res_A['diracc_mean']:.1f}%±{res_A['diracc_std']:.1f}%")

    print("Model B (No supply) 실행 중...")
    res_B = walk_forward_cv(X[feats_B], y, params)
    print(f"  RMSE={res_B['rmse_mean']:.2f}±{res_B['rmse_std']:.2f}, "
          f"DirAcc={res_B['diracc_mean']:.1f}%±{res_B['diracc_std']:.1f}%")

    print("Model C (Supply only) 실행 중...")
    res_C = walk_forward_cv(X[feats_C], y, params)
    print(f"  RMSE={res_C['rmse_mean']:.2f}±{res_C['rmse_std']:.2f}, "
          f"DirAcc={res_C['diracc_mean']:.1f}%±{res_C['diracc_std']:.1f}%")

    # CSV 저장
    results = pd.DataFrame([
        {"model": "A", **res_A},
        {"model": "B", **res_B},
        {"model": "C", **res_C},
    ])
    csv_path = os.path.join(ASML_METRIC_DIR, "asml_ablation_results.csv")
    results.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n결과 저장: {csv_path}")

    # 비교 테이블
    print_comparison(res_A, res_B)

    # SHAP (Model A 전체 데이터로 학습)
    print("SHAP 분석 (Model A, 전체 데이터)...")
    model_full = xgb.XGBRegressor(**params)
    model_full.fit(X[feats_A].values, y.values)
    run_shap(model_full, X[feats_A], "ASML")

    print("완료.")


if __name__ == "__main__":
    main()
