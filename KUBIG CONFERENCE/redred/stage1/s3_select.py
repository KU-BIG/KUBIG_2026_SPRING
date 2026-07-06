"""
s3_select.py — Step 3: 피처 선택
===================================
다중공선성 제거 → XGBoost + SHAP 중요도 → 커스텀 RFE 곡선으로
전체 피처를 20개 안팎으로 압축한다. Hold-out 24개월은 사전 분리해
데이터 누수(leakage)를 방지한다.

입력:  outputs/data/features_dataset.csv
       outputs/models/best_xgboost.pkl  (Step 2 결과, 없으면 기본 파라미터 사용)
출력:  outputs/models/best_xgboost_selected.pkl
       outputs/figures/correlation_heatmap.png
       outputs/figures/xgb_importance.png
       outputs/figures/rfe_curve.png
       outputs/metrics/selected_features.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")

from config import (
    FEATURES_PATH, TUNED_PKL, SELECTED_PKL,
    FIG_DIR, METRIC_DIR,
    PRIMARY_TARGET, TEST_EVAL_SIZE,
    N_SPLITS, TEST_SIZE, MIN_TRAIN, RANDOM_STATE,
    W_BULL_CORRECT, W_BULL_WRONG, W_BEAR_CORRECT, W_BEAR_WRONG, BEAR_SAMPLE_W,
)

CORR_THRESHOLD = 0.9
RFE_ELBOW_TOL  = 0.05
RFE_CANDIDATES = [10, 15, 18, 20, 22, 25, 30, 35, 40, 50, 70]

DEFAULT_XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
)


# ── 데이터 로드 ────────────────────────────────────────────────
def load_and_split():
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    target_col = PRIMARY_TARGET
    if target_col not in df.columns:
        target_col = [c for c in df.columns if c.startswith("TARGET_")][0]
    feature_cols = [c for c in df.columns if not c.startswith("TARGET_")]
    df_clean = df.dropna(subset=[target_col])
    X = df_clean[feature_cols].ffill().fillna(0)
    y = df_clean[target_col]
    split = len(X) - TEST_EVAL_SIZE
    X_tune, X_ho = X.iloc[:split], X.iloc[split:]
    y_tune, y_ho = y.iloc[:split], y.iloc[split:]
    print(f"  전체 {X.shape[0]}개월 × {X.shape[1]}개 피처")
    print(f"  Tune: {len(X_tune)}개월  Holdout: {len(X_ho)}개월")
    return X, y, X_tune, y_tune, X_ho, y_ho


# ── XGBoost 파라미터 로드 ──────────────────────────────────────
def load_xgb_params() -> dict:
    keep_keys = {"n_estimators", "learning_rate", "max_depth", "subsample",
                 "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_weight"}
    if os.path.exists(TUNED_PKL):
        with open(TUNED_PKL, "rb") as f:
            data = pickle.load(f)
        raw = data["model"].get_params()
        params = {k: v for k, v in raw.items() if k in keep_keys and v is not None}
        params.update({"random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1})
        print(f"  best_xgboost.pkl에서 튜닝 파라미터 로드")
        return params
    print("  [주의] best_xgboost.pkl 없음 → 기본 파라미터 사용")
    return DEFAULT_XGB_PARAMS.copy()


# ── Step 2: 다중공선성 제거 ────────────────────────────────────
def remove_multicollinear(X_tune, y_tune):
    print(f"\n[Step 2] 다중공선성 제거 (|corr| ≥ {CORR_THRESHOLD})")
    corr_matrix = X_tune.corr().abs()
    target_corr = X_tune.corrwith(y_tune).abs()
    upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_remove   = set()
    removal_log = []
    for col in upper.columns:
        for feat in upper.index[upper[col] >= CORR_THRESHOLD].tolist():
            if feat in to_remove or col in to_remove:
                continue
            corr_val = corr_matrix.loc[feat, col]
            if target_corr.get(feat, 0) < target_corr.get(col, 0):
                to_remove.add(feat);  removal_log.append((feat, col, corr_val))
            else:
                to_remove.add(col);   removal_log.append((col, feat, corr_val))
    kept = [c for c in X_tune.columns if c not in to_remove]
    print(f"  제거: {len(to_remove)}개  잔존: {len(kept)}개")

    # 히트맵
    n     = min(40, len(target_corr))
    top40 = target_corr[kept].nlargest(n).index.tolist()
    fig_h = max(10, n * 0.38)
    fig, ax = plt.subplots(figsize=(fig_h * 1.1, fig_h))
    sns.heatmap(X_tune[top40].corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=False, linewidths=0.3, ax=ax)
    ax.set_title(f"피처 상관관계 히트맵 (타겟 상관 상위 {n}개)", fontsize=12)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return kept, to_remove, removal_log, target_corr


# ── Bear 가중치 헬퍼 (전체 스크립트 공유) ─────────────────────
def _bear_weights(y) -> np.ndarray:
    return np.where(np.asarray(y) > 0, 1.0, BEAR_SAMPLE_W)


def _asymloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    bull    = y_true > 0
    bear    = ~bull
    correct = (y_true > 0) == (y_pred > 0)
    w = np.where(bull & correct,  W_BULL_CORRECT,
        np.where(bull & ~correct, W_BULL_WRONG,
        np.where(bear & correct,  W_BEAR_CORRECT, W_BEAR_WRONG)))
    return float(np.sqrt((w * (y_true - y_pred) ** 2).sum() / w.sum()))


# ── Step 3: XGBoost 중요도 + SHAP ─────────────────────────────
def extract_importance(X_tune, y_tune, kept, xgb_params):
    print(f"\n[Step 3] XGBoost 중요도 계산 ({len(kept)}개 피처)")
    X_sub = X_tune[kept]
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_sub, y_tune)
    xgb_imp_norm = model.feature_importances_ / (model.feature_importances_.sum() + 1e-12)

    shap_imp_norm  = None
    try:
        import shap
        explainer     = shap.TreeExplainer(model)
        shap_values   = explainer.shap_values(X_sub)
        shap_imp      = np.abs(shap_values).mean(axis=0)
        shap_imp_norm = shap_imp / (shap_imp.sum() + 1e-12)
        print(f"  SHAP 계산 완료")
        shap.summary_plot(shap_values, X_sub, plot_type="bar", max_display=30, show=False)
        fig = plt.gcf()
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("  [SHAP 미설치] XGBoost importance만 사용 (pip install shap 권장)")

    combined = (0.5 * xgb_imp_norm + 0.5 * shap_imp_norm) if shap_imp_norm is not None else xgb_imp_norm

    importance_df = pd.DataFrame({
        "feature":        kept,
        "xgb_importance": xgb_imp_norm,
        "shap_importance": shap_imp_norm if shap_imp_norm is not None else np.nan,
        "combined_score": combined,
    }).sort_values("combined_score", ascending=False).reset_index(drop=True)

    top30 = importance_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["darkorange" if i < 10 else "steelblue" for i in range(len(top30))]
    ax.barh(range(len(top30)), top30["combined_score"].values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"].values[::-1], fontsize=8)
    ax.set_xlabel("통합 중요도 (XGBoost + SHAP, 정규화)")
    ax.set_title("피처 통합 중요도 상위 30개", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "xgb_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return importance_df


# ── Step 4: RFE 곡선 (AsymLoss 기준) ─────────────────────────
def _cv_asymloss(features, X_tune, y_tune, xgb_params):
    """피처 선택 기준: Bear sample_weight 적용 + AsymLoss 평가"""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    scores = []
    X_sub = X_tune[features]
    for tr, te in tscv.split(X_sub):
        if len(tr) < MIN_TRAIN: continue
        w_tr = _bear_weights(y_tune.iloc[tr])
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_sub.iloc[tr], y_tune.iloc[tr], sample_weight=w_tr)
        preds = m.predict(X_sub.iloc[te])
        scores.append(_asymloss(y_tune.iloc[te].values, preds))
    return float(np.mean(scores)) if scores else float("inf")


def run_rfe_curve(importance_df, X_tune, y_tune, xgb_params):
    print(f"\n[Step 4] RFE 곡선 (TimeSeriesSplit {N_SPLITS}-fold, 기준: AsymLoss)")
    ranked    = importance_df["feature"].tolist()
    n_total   = len(ranked)
    candidates = sorted(set(RFE_CANDIDATES + [n_total]))
    candidates = [n for n in candidates if n <= n_total]

    rfe_rows = []
    for n in candidates:
        loss = _cv_asymloss(ranked[:n], X_tune, y_tune, xgb_params)
        rfe_rows.append({"n_features": n, "cv_asymloss": loss})
        print(f"  n={n:3d}  CV AsymLoss: {loss:.4f}")

    rfe_df    = pd.DataFrame(rfe_rows)
    best_loss = rfe_df["cv_asymloss"].min()
    tolerance = best_loss * (1 + RFE_ELBOW_TOL)
    optimal_n = int(rfe_df[rfe_df["cv_asymloss"] <= tolerance]["n_features"].min())
    opt_loss  = float(rfe_df.loc[rfe_df["n_features"] == optimal_n, "cv_asymloss"].iloc[0])
    print(f"\n  ★ 최적 피처 수: {optimal_n}개  CV AsymLoss: {opt_loss:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rfe_df["n_features"], rfe_df["cv_asymloss"], marker="o", color="steelblue", label="CV AsymLoss")
    ax.axvline(optimal_n, color="darkorange", linestyle="--",
               label=f"최적: {optimal_n}개 (AsymLoss={opt_loss:.3f})")
    ax.axhline(tolerance, color="gray", linestyle=":", label=f"+{RFE_ELBOW_TOL*100:.0f}% 허용")
    ax.scatter([optimal_n], [opt_loss], color="darkorange", s=120, zorder=5)
    ax.set_xlabel("피처 수"); ax.set_ylabel("CV AsymLoss")
    ax.set_title("RFE: 피처 수 vs CV AsymLoss (Bear 최적화)", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rfe_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ranked[:optimal_n], rfe_df, optimal_n


# ── Step 5: 최종 재학습 및 selected.pkl 저장 ──────────────────
def final_evaluation(selected, X, y, X_tune, y_tune, X_ho, y_ho, xgb_params):
    print(f"\n[Step 5] 최종 재학습 및 비교")

    def _eval(feats, label):
        w_tune = _bear_weights(y_tune)
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tune[feats], y_tune, sample_weight=w_tune)
        ho_pred = m.predict(X_ho[feats])
        rmse    = float(np.sqrt(mean_squared_error(y_ho.values, ho_pred)))
        dir_acc = float(np.mean((y_ho.values > 0) == (ho_pred > 0))) * 100
        cv_al   = _cv_asymloss(feats, X_tune, y_tune, xgb_params)
        return {"label": label, "n": len(feats), "cv_asymloss": cv_al,
                "holdout_rmse": rmse, "dir_acc": dir_acc}

    full_feats = list(X.columns)
    r_full = _eval(full_feats, f"전체 {len(full_feats)}개")
    r_sel  = _eval(selected,   f"선택 {len(selected)}개")

    print(f"\n  {'':32} {'CV AsymLoss':>12} {'Hold-out':>10} {'DirAcc':>8}")
    print(f"  {'-'*64}")
    for r in [r_full, r_sel]:
        print(f"  {r['label']:<32} {r['cv_asymloss']:>12.4f} "
              f"{r['holdout_rmse']:>10.4f} {r['dir_acc']:>7.1f}%")

    # best_xgboost_selected.pkl 저장
    print(f"\n  전체 {len(X)}개월로 최종 모델 재훈련 중...")
    final_m = xgb.XGBRegressor(**xgb_params)
    final_m.fit(X[selected], y)
    with open(SELECTED_PKL, "wb") as f:
        pickle.dump({"model": final_m, "feature_names": selected,
                     "holdout_rmse": r_sel["holdout_rmse"],
                     "n_features": len(selected)}, f)
    print(f"  → 저장: {SELECTED_PKL}")
    return r_full, r_sel


# ── 피처 요약 CSV ─────────────────────────────────────────────
def save_summary_csv(importance_df, all_feats, to_remove, removal_log,
                     target_corr, selected):
    removal_map = {rm: (kp, cv) for rm, kp, cv in removal_log}
    imp_lk  = importance_df.set_index("feature")
    sel_set = set(selected)
    rows = []
    for feat in all_feats:
        is_rm  = feat in to_remove
        is_sel = feat in sel_set
        tc     = float(target_corr.get(feat, np.nan))
        if is_rm:
            kp, cv = removal_map.get(feat, ("?", float("nan")))
            reason = f"다중공선성 제거 ('{kp}'와 corr={cv:.3f})"
            xi = si = co = np.nan
        elif feat in imp_lk.index:
            row  = imp_lk.loc[feat]
            xi   = float(row["xgb_importance"])
            si   = float(row["shap_importance"]) if not pd.isna(row["shap_importance"]) else np.nan
            co   = float(row["combined_score"])
            rank = int(importance_df[importance_df["feature"] == feat].index[0]) + 1
            reason = f"중요도 {rank}위 선정" if is_sel else f"RFE 탈락 ({rank}위, score={co:.4f})"
        else:
            xi = si = co = np.nan; reason = "알 수 없음"
        rows.append({"feature": feat, "xgb_importance": xi, "shap_importance": si,
                     "combined_score": co, "target_corr": tc,
                     "removed_collinear": is_rm, "selected": is_sel,
                     "selection_reason": reason})

    df_out = pd.DataFrame(rows).sort_values(
        ["selected", "combined_score"], ascending=[False, False]
    ).reset_index(drop=True)
    path = os.path.join(METRIC_DIR, "selected_features.csv")
    df_out.to_csv(path, index=False)
    print(f"  → 피처 요약 저장: {path}")

    sel_df = df_out[df_out["selected"]].reset_index(drop=True)
    print(f"\n  최종 선정 피처 ({len(selected)}개):")
    for i, row in sel_df.iterrows():
        print(f"  {i+1:>3}. {row['feature']}")


# ── 메인 ───────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  Step 3  피처 선택")
    print("=" * 64)

    print("\n[1] 데이터 로드")
    X, y, X_tune, y_tune, X_ho, y_ho = load_and_split()

    xgb_params = load_xgb_params()

    kept, to_remove, removal_log, target_corr = remove_multicollinear(X_tune, y_tune)
    importance_df = extract_importance(X_tune, y_tune, kept, xgb_params)
    selected, rfe_df, optimal_n = run_rfe_curve(importance_df, X_tune, y_tune, xgb_params)
    final_evaluation(selected, X, y, X_tune, y_tune, X_ho, y_ho, xgb_params)
    save_summary_csv(importance_df, list(X.columns), to_remove,
                     removal_log, target_corr, selected)

    print(f"\n  Step 3 완료.  선택 피처: {len(selected)}개 / 전체 {len(X.columns)}개")


if __name__ == "__main__":
    main()
