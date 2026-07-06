"""
asml3_horizon.py — Step 3: ASML Horizon Sensitivity (h=1..12)

Model A: wsts_pred_t6 + 전체 피처
Model B: wsts_pred_t6 제외
Model C: wsts_pred_t6만

각 h에 대해 walk-forward CV를 실행하고 ΔRMSE, ΔDirAcc를 계산한다.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asml_config import (
    ASML_FEATURES_PATH, SUPPLY_FINAL_PKL, ASML_FIG_DIR, ASML_METRIC_DIR,
    TARGET_COL, SUPPLY_COL, FIRM_TICKER, MIN_TRAIN_M, RANDOM_STATE,
)


def load_xgb_params() -> dict:
    import pickle
    try:
        with open(SUPPLY_FINAL_PKL, "rb") as f:
            saved = pickle.load(f)
        p = {**saved["best_params"], "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
        print(f"Stage 1 best_params 로드 완료")
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


def load_firm_price() -> pd.Series:
    raw = yf.download(FIRM_TICKER, start="1993-01-01", auto_adjust=True, progress=False)["Close"]
    price_m = raw.resample("ME").last()
    if isinstance(price_m, pd.DataFrame):
        price_m = price_m.squeeze()
    return price_m.dropna()


def main() -> None:
    print("=" * 55)
    print(f"  asml3_horizon.py — ASML Horizon Sensitivity (h=1~12)")
    print("=" * 55)

    df_feat = pd.read_parquet(ASML_FEATURES_PATH)
    feature_cols = [c for c in df_feat.columns if c != TARGET_COL]
    feats_B = [c for c in feature_cols if c != SUPPLY_COL]
    feats_C = [SUPPLY_COL]
    X_base  = df_feat[feature_cols]

    print(f"피처 매트릭스: {df_feat.shape}, "
          f"{df_feat.index[0].strftime('%Y-%m')} ~ {df_feat.index[-1].strftime('%Y-%m')}")
    print(f"피처 A={len(feature_cols)}, B={len(feats_B)}, C={len(feats_C)}")

    price_m = load_firm_price()
    print(f"{FIRM_TICKER} 주가: {price_m.shape[0]}행, "
          f"{price_m.index[0].strftime('%Y-%m')} ~ {price_m.index[-1].strftime('%Y-%m')}")

    params = load_xgb_params()

    records = []
    for h in range(1, 13):
        print(f"\n[h={h}] {FIRM_TICKER}_fwd{h} 계산 중...")
        target_h = price_m.pct_change(h).shift(-h) * 100
        target_h.name = f"{FIRM_TICKER}_fwd{h}"

        df_h = X_base.join(target_h, how="inner").dropna()
        y_h  = df_h[f"{FIRM_TICKER}_fwd{h}"]
        n_folds = len(df_h) - MIN_TRAIN_M

        print(f"  데이터: {df_h.shape}, {n_folds} folds")

        res_A = walk_forward_cv(df_h[feature_cols], y_h, params)
        res_B = walk_forward_cv(df_h[feats_B],      y_h, params)
        res_C = walk_forward_cv(df_h[feats_C],      y_h, params)

        delta_rmse   = res_A["rmse_mean"]   - res_B["rmse_mean"]
        delta_diracc = res_A["diracc_mean"] - res_B["diracc_mean"]

        print(f"  A: RMSE={res_A['rmse_mean']:.2f}, DirAcc={res_A['diracc_mean']:.1f}%")
        print(f"  B: RMSE={res_B['rmse_mean']:.2f}, DirAcc={res_B['diracc_mean']:.1f}%")
        print(f"  C: RMSE={res_C['rmse_mean']:.2f}, DirAcc={res_C['diracc_mean']:.1f}%")
        print(f"  Δ(A-B): RMSE={delta_rmse:+.3f}, DirAcc={delta_diracc:+.1f}%p")

        records.append({
            "h": h,
            "rmse_a": res_A["rmse_mean"], "rmse_b": res_B["rmse_mean"], "rmse_c": res_C["rmse_mean"],
            "diracc_a": res_A["diracc_mean"], "diracc_b": res_B["diracc_mean"], "diracc_c": res_C["diracc_mean"],
            "delta_rmse":   delta_rmse,
            "delta_diracc": delta_diracc,
            "n_folds": n_folds,
        })

    results = pd.DataFrame(records).set_index("h")

    best_rmse_h   = int(results["delta_rmse"].idxmin())
    best_diracc_h = int(results["delta_diracc"].idxmax())
    print(f"\n최적 h (ΔRMSE 최소):   h={best_rmse_h} (Δ={results.loc[best_rmse_h, 'delta_rmse']:+.3f})")
    print(f"최적 h (ΔDirAcc 최대): h={best_diracc_h} (Δ={results.loc[best_diracc_h, 'delta_diracc']:+.1f}%p)")

    csv_path = os.path.join(ASML_METRIC_DIR, "asml_horizon_results.csv")
    results.reset_index().to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n결과 저장: {csv_path}")

    hs = list(range(1, 13))
    bar_colors_rmse = [
        "tomato"    if h == best_rmse_h else
        "lightgray" if h == 6           else
        "steelblue"
        for h in hs
    ]
    bar_colors_diracc = [
        "tomato"    if h == best_diracc_h else
        "lightgray" if h == 6             else
        "steelblue"
        for h in hs
    ]

    # ΔRMSE bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hs, results["delta_rmse"].values, color=bar_colors_rmse, edgecolor="white", width=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Forecast Horizon h (months)")
    ax.set_ylabel("ΔRMSE (Model A − Model B)")
    ax.set_title("ASML: Supply Signal Contribution by Horizon (RMSE)")
    ax.set_xticks(hs)
    for h in sorted({best_rmse_h, 6}):
        v = results.loc[h, "delta_rmse"]
        ax.text(h, v - 0.05 if v < 0 else v + 0.03, f"{v:+.2f}",
                ha="center", va="top" if v < 0 else "bottom", fontsize=8)
    plt.tight_layout()
    path_rmse = os.path.join(ASML_FIG_DIR, "asml_horizon_rmse.png")
    plt.savefig(path_rmse, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {path_rmse}")

    # ΔDirAcc bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hs, results["delta_diracc"].values, color=bar_colors_diracc, edgecolor="white", width=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Forecast Horizon h (months)")
    ax.set_ylabel("ΔDirAcc (Model A − Model B, %p)")
    ax.set_title("ASML: Supply Signal Contribution by Horizon (Direction Accuracy)")
    ax.set_xticks(hs)
    for h in sorted({best_diracc_h, 6}):
        v = results.loc[h, "delta_diracc"]
        ax.text(h, v + 0.2 if v >= 0 else v - 0.2, f"{v:+.1f}%p",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    plt.tight_layout()
    path_diracc = os.path.join(ASML_FIG_DIR, "asml_horizon_diracc.png")
    plt.savefig(path_diracc, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {path_diracc}")

    print("\n완료.")


if __name__ == "__main__":
    main()
