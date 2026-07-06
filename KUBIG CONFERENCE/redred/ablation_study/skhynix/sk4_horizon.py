"""
sk4_horizon.py — Horizon Sensitivity Analysis

wsts_pred_t6가 h개월 앞의 SK하이닉스 주가 수익률 예측에 얼마나 기여하는지
h=1..12에 대해 Ablation(A/B/C)을 반복해 최적 horizon을 찾는다.

Model A (Full):        wsts_pred_t6 + macro + SK + FX  (12개 피처)
Model B (No Supply):   macro + SK + FX                 (11개 피처)
Model C (Supply Only): wsts_pred_t6                    (1개 피처)
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from sk_config import (
    STAGE2_PATH,
    PRICE_PATH,
    SUPPLY_FINAL_PKL,
    SK_FIG_DIR,
    SK_METRIC_DIR,
    MIN_TRAIN_M,
    RANDOM_STATE,
)

HORIZONS = list(range(1, 13))
SUPPLY_COL = "wsts_pred_t6"


# ── XGBoost 파라미터 로드 ──────────────────────────────────────────────────────

def load_xgb_params() -> dict:
    with open(SUPPLY_FINAL_PKL, "rb") as f:
        stage1 = pickle.load(f)
    # pkl은 {'model': ..., 'feature_names': ..., 'best_params': {...}} 구조
    params = dict(stage1["best_params"])
    params.update({"random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1})
    return params


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv(X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    n = len(X)
    rmse_list, diracc_list = [], []
    for test_end in range(MIN_TRAIN_M, n):
        X_tr = X.iloc[:test_end].values
        y_tr = y.iloc[:test_end].values
        X_te = X.iloc[test_end : test_end + 1].values
        y_te = y.iloc[test_end : test_end + 1].values

        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)

        rmse_list.append(float(np.sqrt(mean_squared_error(y_te, pred))))
        diracc_list.append(float(((y_te > 0) == (pred > 0)).mean() * 100))

    return {
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "diracc_mean": float(np.mean(diracc_list)),
        "diracc_std": float(np.std(diracc_list)),
    }


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_metric(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: str,
    hline: float | None = None,
) -> None:
    hs = results["h"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hs, results[f"{metric}_A"], marker="o", label="Model A (Full)", color="#2196F3")
    ax.plot(hs, results[f"{metric}_B"], marker="o", label="Model B (No Supply)", color="#FF9800")
    ax.plot(hs, results[f"{metric}_C"], marker="o", label="Model C (Supply Only)", color="#4CAF50")
    if hline is not None:
        ax.axhline(hline, linestyle="--", color="gray", linewidth=0.8, label=f"y={hline}")
    ax.set_xlabel("Horizon h (months)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(SK_FIG_DIR, exist_ok=True)
    os.makedirs(SK_METRIC_DIR, exist_ok=True)

    params = load_xgb_params()
    print(f"XGBoost params 로드 완료: n_estimators={params['n_estimators']}, "
          f"lr={params['learning_rate']:.4f}, max_depth={params['max_depth']}")

    # 피처 로드
    df_feat = pd.read_parquet(STAGE2_PATH)
    feature_cols = [c for c in df_feat.columns if c != "hynix_fwd6"]
    df_feat = df_feat[feature_cols]

    # 주가 시계열 로드
    price_series = pd.read_parquet(PRICE_PATH)["hynix_price"]

    feats_A = feature_cols
    feats_B = [c for c in feature_cols if c != SUPPLY_COL]
    feats_C = [SUPPLY_COL]

    print(f"\n피처 수: A={len(feats_A)}, B={len(feats_B)}, C={len(feats_C)}")
    print(f"stage2_features: {len(df_feat)}행, {df_feat.index[0].strftime('%Y-%m')} ~ {df_feat.index[-1].strftime('%Y-%m')}")
    print(f"hynix_price: {len(price_series)}행, {price_series.index[0].strftime('%Y-%m')} ~ {price_series.index[-1].strftime('%Y-%m')}")
    print("\n" + "=" * 60)
    print(f"  Horizon Sensitivity 실험 시작 (h=1..12)")
    print("=" * 60)

    records = []
    for h in HORIZONS:
        # h개월 선행 수익률 타겟 계산
        target_h = price_series.pct_change(h).shift(-h) * 100
        target_h.name = f"hynix_fwd{h}"

        df = df_feat.join(target_h, how="inner").dropna()
        X = df[feature_cols]
        y = df[f"hynix_fwd{h}"]
        n_folds = len(X) - MIN_TRAIN_M

        print(f"\nh={h:2d}: {len(df)}행, {n_folds} folds — 실험 중...")

        res_A = walk_forward_cv(X[feats_A], y, params)
        res_B = walk_forward_cv(X[feats_B], y, params)
        res_C = walk_forward_cv(X[feats_C], y, params)

        delta_rmse = res_A["rmse_mean"] - res_B["rmse_mean"]   # - → A가 더 좋음 (A RMSE < B RMSE)
        delta_diracc = res_A["diracc_mean"] - res_B["diracc_mean"]  # + → A가 더 좋음

        records.append({
            "h": h,
            "rmse_A": res_A["rmse_mean"],
            "rmse_B": res_B["rmse_mean"],
            "rmse_C": res_C["rmse_mean"],
            "diracc_A": res_A["diracc_mean"],
            "diracc_B": res_B["diracc_mean"],
            "diracc_C": res_C["diracc_mean"],
            "delta_rmse": delta_rmse,
            "delta_diracc": delta_diracc,
        })

        print(f"  A: RMSE={res_A['rmse_mean']:.2f}, DirAcc={res_A['diracc_mean']:.1f}%")
        print(f"  B: RMSE={res_B['rmse_mean']:.2f}, DirAcc={res_B['diracc_mean']:.1f}%")
        print(f"  C: RMSE={res_C['rmse_mean']:.2f}, DirAcc={res_C['diracc_mean']:.1f}%")
        print(f"  Δ(A-B): ΔRMSE={delta_rmse:+.2f}, ΔDirAcc={delta_diracc:+.1f}%p")

    # 결과 데이터프레임
    results = pd.DataFrame(records)

    # 요약 출력
    print("\n" + "─" * 52)
    print("  Horizon Sensitivity: wsts_pred_t6 기여도 요약")
    print("─" * 52)
    for _, row in results.iterrows():
        print(f"  h={int(row['h']):2d}:  ΔRMSE={row['delta_rmse']:+6.2f}  "
              f"ΔDirAcc={row['delta_diracc']:+5.1f}%p")
    print("─" * 52)

    best_diracc_h = int(results.loc[results["delta_diracc"].idxmax(), "h"])
    best_diracc_val = results["delta_diracc"].max()
    # delta_rmse = rmse_A - rmse_B → 음수일수록 A가 더 좋음 → idxmin
    best_rmse_h = int(results.loc[results["delta_rmse"].idxmin(), "h"])
    best_rmse_val = results["delta_rmse"].min()

    print(f"  최적 horizon (DirAcc 기준): h={best_diracc_h}  (ΔDirAcc={best_diracc_val:+.1f}%p)")
    print(f"  최적 horizon (RMSE 기준):   h={best_rmse_h}  (ΔRMSE={best_rmse_val:+.2f})")
    print("─" * 52)

    # CSV 저장
    csv_path = os.path.join(SK_METRIC_DIR, "sk_horizon_results.csv")
    results.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n결과 저장: {csv_path}")

    # 그래프 저장
    print("\n그래프 저장 중...")
    plot_metric(
        results,
        metric="diracc",
        ylabel="Direction Accuracy (%)",
        title="Horizon Sensitivity: Direction Accuracy by h (A/B/C)",
        out_path=os.path.join(SK_FIG_DIR, "sk_horizon_diracc.png"),
        hline=50.0,
    )
    plot_metric(
        results,
        metric="rmse",
        ylabel="RMSE",
        title="Horizon Sensitivity: RMSE by h (A/B/C)",
        out_path=os.path.join(SK_FIG_DIR, "sk_horizon_rmse.png"),
    )

    print("\n완료.")


if __name__ == "__main__":
    main()
