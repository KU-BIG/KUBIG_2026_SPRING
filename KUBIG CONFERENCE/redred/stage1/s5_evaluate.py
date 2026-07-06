"""
s5_evaluate.py — Step 5: 최종 평가 + 시각화
=============================================
best_xgboost_final.pkl을 로드해 7개 표준 지표를 TimeSeriesSplit CV로 계산하고
3종 시각화를 생성한다.

지표:
  1. RMSE (전체 / Bull / Bear)
  2. Direction Accuracy (전체 / Bull / Bear)
  3. Asymmetric Loss

시각화:
  01_prediction_timeline.png  — 전체 기간 예측 vs 실제 + Hold-out 확대 + Bull/Bear 방향 정확도
  02_cv_metrics.png           — 7개 지표 CV 평균 vs Hold-out 바차트
  03_bear_improvement.png     — Baseline(RMSE 목적) vs Final(AsymLoss 목적) 비교

입력:  outputs/data/features_dataset.csv
       outputs/models/best_xgboost_final.pkl
출력:  outputs/figures/*.png
       outputs/metrics/final_cv_metrics.csv
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")

from config import (
    FEATURES_PATH, FINAL_PKL,
    FIG_DIR, METRIC_DIR,
    PRIMARY_TARGET, TEST_EVAL_SIZE,
    N_SPLITS, TEST_SIZE, MIN_TRAIN,
    W_BULL_CORRECT, W_BULL_WRONG, W_BEAR_CORRECT, W_BEAR_WRONG,
    BEAR_SAMPLE_W,
)

# Baseline 수치 (Step 2 RMSE 최적화 모델 CV 평균 — 비교용)
BASELINE = {
    "rmse":      7.312,
    "rmse_bull": 6.278,
    "rmse_bear": 7.364,
    "dir_acc":   90.0,
    "dir_bull":  98.3,
    "dir_bear":  80.5,
    "asym_loss": 7.776,
}

METRIC_KEYS = ["rmse", "rmse_bull", "rmse_bear",
               "dir_acc", "dir_bull", "dir_bear", "asym_loss"]
METRIC_LABELS = {
    "rmse":      "RMSE\n(전체)",
    "rmse_bull": "RMSE\n(Bull)",
    "rmse_bear": "RMSE\n(Bear)",
    "dir_acc":   "DirAcc\n(전체,%)",
    "dir_bull":  "DirAcc\n(Bull,%)",
    "dir_bear":  "DirAcc\n(Bear,%)",
    "asym_loss": "Asymmetric\nLoss",
}
DIR_METRICS = {"dir_acc", "dir_bull", "dir_bear"}


# ── 데이터 / 모델 로드 ─────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    target_col = PRIMARY_TARGET
    if target_col not in df.columns:
        target_col = [c for c in df.columns if c.startswith("TARGET_")][0]
    feature_cols = [c for c in df.columns if not c.startswith("TARGET_")]
    df_clean = df.dropna(subset=[target_col])
    X = df_clean[feature_cols].ffill().fillna(0)
    y = df_clean[target_col]
    split = len(X) - TEST_EVAL_SIZE
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def load_model():
    with open(FINAL_PKL, "rb") as f:
        data = pickle.load(f)
    print(f"  최종 모델 로드: 선택 피처 {len(data['feature_names'])}개")
    return data["model"], data["feature_names"], data.get("best_params", {})


# ── 7개 지표 계산 ──────────────────────────────────────────────
def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bull    = y_true > 0
    bear    = ~bull
    correct = (y_true > 0) == (y_pred > 0)
    w = np.where(bull & correct,  W_BULL_CORRECT,
        np.where(bull & ~correct, W_BULL_WRONG,
        np.where(bear & correct,  W_BEAR_CORRECT, W_BEAR_WRONG)))

    def safe_rmse(m): return float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))) if m.any() else None
    def safe_dir(m):  return float(correct[m].mean() * 100) if m.any() else None

    return {
        "rmse":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmse_bull": safe_rmse(bull),
        "rmse_bear": safe_rmse(bear),
        "dir_acc":   float(correct.mean() * 100),
        "dir_bull":  safe_dir(bull),
        "dir_bear":  safe_dir(bear),
        "asym_loss": float(np.sqrt((w * (y_true - y_pred) ** 2).sum() / w.sum())),
    }


def _avg(fold_results, key):
    vals = [r[key] for r in fold_results if r[key] is not None]
    return float(np.mean(vals)) if vals else None


# ── CV 평가 ────────────────────────────────────────────────────
def run_cv_evaluation(model, features, X_tune, y_tune):
    params = model.get_params()
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    X_sub  = X_tune[features]
    fold_results = []
    for fold_i, (tr, te) in enumerate(tscv.split(X_sub), 1):
        if len(tr) < MIN_TRAIN: continue
        w_tr = np.where(y_tune.iloc[tr].values > 0, 1.0, BEAR_SAMPLE_W)
        m = xgb.XGBRegressor(**params)
        m.fit(X_sub.iloc[tr], y_tune.iloc[tr], sample_weight=w_tr)
        preds  = m.predict(X_sub.iloc[te])
        y_t    = y_tune.iloc[te].values
        result = compute_all_metrics(y_t, preds)
        result["fold"]   = f"Fold {fold_i}"
        result["period"] = (f"{y_tune.index[te[0]].strftime('%Y-%m')} ~ "
                            f"{y_tune.index[te[-1]].strftime('%Y-%m')}")
        result["tune_preds_idx"]  = te
        result["tune_preds_vals"] = preds
        fold_results.append(result)
    return fold_results


# ── Hold-out 평가 ──────────────────────────────────────────────
def evaluate_holdout(model, features, X_tune, y_tune, X_ho, y_ho):
    params = model.get_params()
    w_tune = np.where(y_tune.values > 0, 1.0, BEAR_SAMPLE_W)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tune[features], y_tune, sample_weight=w_tune)
    preds   = m.predict(X_ho[features])
    metrics = compute_all_metrics(y_ho.values, preds)
    metrics["fold"]   = "Hold-out"
    metrics["period"] = (f"{y_ho.index[0].strftime('%Y-%m')} ~ "
                         f"{y_ho.index[-1].strftime('%Y-%m')}")
    return metrics, preds


# ── 결과 출력 ──────────────────────────────────────────────────
def print_results(fold_results, avg_metrics, holdout_metrics):
    def _f(v, key):
        if v is None: return "  N/A "
        return f"{v:.1f}%" if key in DIR_METRICS else f"{v:.3f}"

    header = (f"  {'구간':<22} {'RMSE':>7} {'Bull':>7} {'Bear':>7} "
              f"{'DirAcc':>8} {'DBull':>7} {'DBear':>7} {'AsymLoss':>9}")
    sep    = "  " + "-" * (len(header) - 2)
    print(header); print(sep)
    for r in fold_results:
        print(f"  {r['fold']:<22} "
              f"{_f(r['rmse'],'rmse'):>7} {_f(r['rmse_bull'],'rmse'):>7} "
              f"{_f(r['rmse_bear'],'rmse'):>7} "
              f"{_f(r['dir_acc'],'dir_acc'):>8} {_f(r['dir_bull'],'dir_bull'):>7} "
              f"{_f(r['dir_bear'],'dir_bear'):>7} {_f(r['asym_loss'],'asym_loss'):>9}")
    print(sep)
    r = avg_metrics
    print(f"  {'CV 평균':<22} "
          f"{_f(r['rmse'],'rmse'):>7} {_f(r['rmse_bull'],'rmse'):>7} "
          f"{_f(r['rmse_bear'],'rmse'):>7} "
          f"{_f(r['dir_acc'],'dir_acc'):>8} {_f(r['dir_bull'],'dir_bull'):>7} "
          f"{_f(r['dir_bear'],'dir_bear'):>7} {_f(r['asym_loss'],'asym_loss'):>9}")
    print(sep)
    r = holdout_metrics
    print(f"  {'Hold-out (OOS)':<22} "
          f"{_f(r['rmse'],'rmse'):>7} {_f(r['rmse_bull'],'rmse'):>7} "
          f"{_f(r['rmse_bear'],'rmse'):>7} "
          f"{_f(r['dir_acc'],'dir_acc'):>8} {_f(r['dir_bull'],'dir_bull'):>7} "
          f"{_f(r['dir_bear'],'dir_bear'):>7} {_f(r['asym_loss'],'asym_loss'):>9}")


# ── CSV 저장 ───────────────────────────────────────────────────
def save_csv(fold_results, avg_metrics, holdout_metrics):
    cols = ["fold", "period"] + METRIC_KEYS
    rows = [{k: r.get(k) for k in cols} for r in fold_results]
    rows.append({**{k: avg_metrics.get(k) for k in METRIC_KEYS},
                 "fold": "CV 평균", "period": "-"})
    rows.append({**{k: holdout_metrics.get(k) for k in METRIC_KEYS},
                 "fold": "Hold-out", "period": holdout_metrics.get("period", "-")})
    df = pd.DataFrame(rows, columns=cols)
    path = os.path.join(METRIC_DIR, "final_cv_metrics.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → CSV 저장: {path}")


# ──────────────────────────────────────────────────────────────
# Figure 1: 전체 예측 타임라인 (3-panel)
# ──────────────────────────────────────────────────────────────
def fig1_prediction_timeline(model, features, X_tune, y_tune, X_ho, y_ho):
    params = model.get_params()
    w_tune = np.where(y_tune.values > 0, 1.0, BEAR_SAMPLE_W)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tune[features], y_tune, sample_weight=w_tune)

    tune_preds    = m.predict(X_tune[features])
    holdout_preds = m.predict(X_ho[features])

    y_all         = pd.concat([y_tune, y_ho])
    preds_all     = np.concatenate([tune_preds, holdout_preds])
    holdout_start = y_ho.index[0]

    bull_ho  = y_ho.values > 0
    bear_ho  = ~bull_ho
    correct  = (y_ho.values > 0) == (holdout_preds > 0)
    ho_bull_acc = float(correct[bull_ho].mean() * 100) if bull_ho.any() else None
    ho_bear_acc = float(correct[bear_ho].mean() * 100) if bear_ho.any() else None
    ho_dir_acc  = float(correct.mean() * 100)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             gridspec_kw={"height_ratios": [2, 1.3, 1.0]})

    # ① 전체 기간
    ax = axes[0]
    ax.plot(y_all.index, y_all.values, label="실제값", color="steelblue", linewidth=2.0)
    ax.plot(y_all.index, preds_all, label=f"예측값 ({len(features)}개 피처)",
            color="darkorange", linewidth=1.6, linestyle="--", alpha=0.9)
    ax.axvline(holdout_start, color="crimson", linewidth=1.8, linestyle=":",
               label=f"Hold-out 시작 ({holdout_start.strftime('%Y-%m')})")
    ax.axvspan(holdout_start, y_all.index[-1], alpha=0.07, color="crimson")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    def _fmt(v): return f"{v:.1f}%" if v is not None else "N/A"
    info = (f"Hold-out  DirAcc={ho_dir_acc:.1f}%  "
            f"Bull={_fmt(ho_bull_acc)}  Bear={_fmt(ho_bear_acc)}")
    ax.text(0.01, 0.03, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88))
    ax.set_title(f"최종 모델 예측 vs 실제값  (선택 피처 {len(features)}개, Bear 최적화)", fontsize=13)
    ax.set_ylabel("Worldwide YoY (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # ② Hold-out 확대
    ax2 = axes[1]
    ax2.plot(y_ho.index, y_ho.values, label="실제값",
             color="steelblue", linewidth=2.2, marker="o", markersize=6)
    ax2.plot(y_ho.index, holdout_preds, label="예측값",
             color="darkorange", linewidth=1.8, linestyle="--", marker="^", markersize=6)
    ax2.fill_between(y_ho.index, y_ho.values, holdout_preds,
                     alpha=0.13, color="crimson", label="예측 오차")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
    for date, act, pred in zip(y_ho.index, y_ho.values, holdout_preds):
        ax2.axvline(date, color="green" if (act > 0) == (pred > 0) else "red",
                    linewidth=0.5, alpha=0.3)
    ho_rmse = float(np.sqrt(mean_squared_error(y_ho.values, holdout_preds)))
    ax2.set_title(f"Hold-out 구간 확대 — RMSE={ho_rmse:.3f} | DirAcc={ho_dir_acc:.1f}%", fontsize=11)
    ax2.set_ylabel("Worldwide YoY (%)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ③ Bull/Bear 방향 정확도 (Tune CV 평균 추정치와 Hold-out 비교)
    ax3 = axes[2]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    cv_bull_accs, cv_bear_accs = [], []
    X_sub = X_tune[features]
    for tr, te in tscv.split(X_sub):
        if len(tr) < MIN_TRAIN: continue
        m2 = xgb.XGBRegressor(**params)
        m2.fit(X_sub.iloc[tr], y_tune.iloc[tr],
               sample_weight=np.where(y_tune.iloc[tr].values > 0, 1.0, BEAR_SAMPLE_W))
        p   = m2.predict(X_sub.iloc[te])
        y_t = y_tune.iloc[te].values
        c_t = (y_t > 0) == (p > 0)
        bm  = y_t > 0; rm = ~bm
        cv_bull_accs.append(float(c_t[bm].mean() * 100) if bm.any() else None)
        cv_bear_accs.append(float(c_t[rm].mean() * 100) if rm.any() else None)

    def nanmean(lst): return float(np.mean([v for v in lst if v is not None])) if any(v is not None for v in lst) else None
    cv_bull = nanmean(cv_bull_accs)
    cv_bear = nanmean(cv_bear_accs)

    x     = np.arange(2)
    width = 0.32
    bull_vals = [cv_bull if cv_bull is not None else 0, ho_bull_acc if ho_bull_acc is not None else 0]
    bear_vals = [cv_bear if cv_bear is not None else 0, ho_bear_acc if ho_bear_acc is not None else 0]
    b1 = ax3.bar(x - width / 2, bull_vals, width, label="Bull Acc", color="#2ecc71", alpha=0.85)
    b2 = ax3.bar(x + width / 2, bear_vals, width, label="Bear Acc", color="#e74c3c", alpha=0.85)
    for bar, v in zip(b1, bull_vals):
        lbl = f"{v:.1f}%" if v > 0 else "N/A"
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(b2, bear_vals):
        lbl = f"{v:.1f}%" if v > 0 else "N/A"
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.axhline(50, color="gray", linewidth=1.0, linestyle="--", alpha=0.6, label="50% (랜덤 수준)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["CV 평균 (in-sample)", "Hold-out (out-of-sample)"], fontsize=10)
    ax3.set_ylabel("Direction Accuracy (%)")
    ax3.set_ylim(0, 115)
    ax3.set_title("Bull / Bear 구간별 방향 정확도", fontsize=12)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(h_pad=3)
    path = os.path.join(FIG_DIR, "01_prediction_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure 1 저장: {path}")


# ──────────────────────────────────────────────────────────────
# Figure 2: CV 지표 요약 바차트
# ──────────────────────────────────────────────────────────────
def fig2_cv_metrics(avg_metrics, holdout_metrics):
    labels    = [METRIC_LABELS[k] for k in METRIC_KEYS]
    cv_vals   = [avg_metrics.get(k) or 0 for k in METRIC_KEYS]
    ho_vals   = [holdout_metrics.get(k) or 0 for k in METRIC_KEYS]

    x     = np.arange(len(METRIC_KEYS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    b1 = ax.bar(x - width / 2, cv_vals, width, label="CV 평균 (5-fold)", color="#3498db", alpha=0.85)
    b2 = ax.bar(x + width / 2, ho_vals, width, label="Hold-out (OOS)",   color="#e67e22", alpha=0.85)

    def _lbl(v, key): return f"{v:.1f}%" if key in DIR_METRICS else f"{v:.3f}"
    for bar, v, k in zip(b1, cv_vals, METRIC_KEYS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                _lbl(v, k), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2980b9")
    for bar, v, k in zip(b2, ho_vals, METRIC_KEYS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                _lbl(v, k), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#d35400")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("값 (RMSE·AsymLoss: 낮을수록 / DirAcc: 높을수록 우수)")
    ax.set_title("표준 평가 지표: CV 평균 vs Hold-out (최종 Bear 최적화 모델)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "02_cv_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure 2 저장: {path}")


# ──────────────────────────────────────────────────────────────
# Figure 3: Baseline vs Final 비교
# ──────────────────────────────────────────────────────────────
def fig3_bear_improvement(avg_metrics):
    labels    = [METRIC_LABELS[k] for k in METRIC_KEYS]
    base_vals = [BASELINE.get(k, 0) or 0 for k in METRIC_KEYS]
    new_vals  = [avg_metrics.get(k) or 0 for k in METRIC_KEYS]

    x     = np.arange(len(METRIC_KEYS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    b1 = ax.bar(x - width / 2, base_vals, width, label="Baseline (RMSE 목적함수)", color="#95a5a6", alpha=0.85)
    b2 = ax.bar(x + width / 2, new_vals,  width, label="Final (AsymLoss + Bear 가중치)", color="#e74c3c", alpha=0.85)

    def _lbl(v, key): return f"{v:.1f}%" if key in DIR_METRICS else f"{v:.3f}"
    for bar, v, k in zip(b1, base_vals, METRIC_KEYS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                _lbl(v, k), ha="center", va="bottom", fontsize=8, color="#555")
    for bar, v, k in zip(b2, new_vals, METRIC_KEYS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                _lbl(v, k), ha="center", va="bottom", fontsize=8, fontweight="bold", color="#c0392b")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("값 (RMSE·AsymLoss: 낮을수록 / DirAcc: 높을수록 우수)")
    ax.set_title("Bear 최적화 효과: Baseline vs Final 모델 — CV 평균 7개 지표 비교", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "03_bear_improvement.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure 3 저장: {path}")


# ── 피처 설명 ──────────────────────────────────────────────────
_REGION = {
    "Worldwide":    "전 세계",
    "Asia_Pacific": "아시아-태평양",
    "Americas":     "아메리카",
    "Europe":       "유럽",
    "Japan":        "일본",
}
_FRED = {
    "SemiProd":  "반도체 산업생산지수",
    "ISM_Mfg":   "ISM 제조업 PMI",
    "T10Y2Y":    "10Y-2Y 금리차",
    "IndProd":   "산업생산지수",
    "PCE_Core":  "근원 PCE 인플레이션",
    "MfgEmp":    "제조업 고용자 수",
    "ConsSenti": "미시간대 소비자 심리지수",
    "NewOrder":  "신규 제조업 수주",
}
_TICKER = {
    "SOX":     "필라델피아 반도체 지수",
    "NVDA":    "NVIDIA",
    "TSM":     "TSMC",
    "ASML":    "ASML",
    "Samsung": "삼성전자",
    "SKHynix": "SK하이닉스",
}
_SUFFIX = {
    "lag6":          "6개월 전 값",
    "lag12":         "12개월 전 값",
    "ma3":           "3개월 이동평균",
    "ma6":           "6개월 이동평균",
    "ma12":          "12개월 이동평균",
    "vol3":          "3개월 변동성(표준편차)",
    "vol6":          "6개월 변동성(표준편차)",
    "momentum_3_12": "단기(3M)-장기(12M) 모멘텀",
    "accel":         "전월 대비 가속도",
    "vs_ma24":       "24개월 구간 내 상대 위치(0~1)",
    "chg3":          "3개월 변화량",
    "chg6":          "6개월 변화량",
    "diff3":         "3개월 변화량",
    "diff6":         "6개월 변화량",
    "diff12":        "12개월 변화량",
}
_EXACT = {
    "InvSales":          "재고/매출 비율 [ISRATIO] — 수준값 (재고 축적 신호)",
    "InvSales_diff3":    "재고/매출 비율 — 3개월 변화량",
    "InvSales_diff6":    "재고/매출 비율 — 6개월 변화량",
    "InvSales_lag6":     "재고/매출 비율 — 6개월 전 값",
    "InvSales_lag12":    "재고/매출 비율 — 12개월 전 값",
    "T10Y3M":            "10Y-3M 금리차 [T10Y3M] — 수준값 (역전 = 경기침체 선행)",
    "T10Y3M_chg3":       "10Y-3M 금리차 — 3개월 변화량",
    "T10Y3M_chg6":       "10Y-3M 금리차 — 6개월 변화량",
    "T10Y3M_inverted":   "장단기 금리 역전 여부 더미 (역전=1)",
    "T10Y3M_inv_streak": "장단기 금리 역전 연속 기간 (개월 수)",
    "T10Y3M_lag6":       "10Y-3M 금리차 — 6개월 전 값",
    "T10Y3M_lag12":      "10Y-3M 금리차 — 12개월 전 값",
    "FedFunds":          "연방기금금리 [DFF] — 수준값 (통화긴축 사이클)",
    "FedFunds_diff6":    "연방기금금리 — 6개월 변화량",
    "FedFunds_diff12":   "연방기금금리 — 12개월 변화량",
    "FedFunds_lag6":     "연방기금금리 — 6개월 전 값",
    "FedFunds_lag12":    "연방기금금리 — 12개월 전 값",
    "FRED_T10Y2Y":       "10Y-2Y 금리차 — 수준값",
    "FRED_T10Y2Y_chg3":  "10Y-2Y 금리차 — 3개월 변화량",
    "ISM_above50":       "ISM PMI > 50 더미 (제조업 확장 구간)",
    "ISM_mom3":          "ISM PMI — 3개월 모멘텀",
    "Eq_AvgRet":         "6개 반도체주 평균 월간 수익률",
    "Eq_AvgRet_lag6":    "6개 반도체주 평균 월간 수익률 — 6개월 전 값",
    "Eq_AvgRet_lag12":   "6개 반도체주 평균 월간 수익률 — 12개월 전 값",
    "month_sin":         "계절성 인코딩 — sin(2π × month/12)",
    "month_cos":         "계절성 인코딩 — cos(2π × month/12)",
}


def _describe(name: str) -> str:
    if name in _EXACT:
        return _EXACT[name]

    # {Region}_YoY{_suffix}
    for region, rname in _REGION.items():
        prefix = f"{region}_YoY"
        if name == prefix:
            return f"{rname} 반도체 매출 YoY%"
        if name.startswith(prefix + "_"):
            sfx = name[len(prefix) + 1:]
            return f"{rname} 반도체 매출 YoY% — {_SUFFIX.get(sfx, sfx)}"

    # FRED_{Series}_YoY{_suffix}
    if name.startswith("FRED_"):
        rest = name[5:]
        for series, sname in _FRED.items():
            yp = f"{series}_YoY"
            if rest == yp:
                return f"{sname} YoY%"
            if rest.startswith(yp + "_"):
                sfx = rest[len(yp) + 1:]
                return f"{sname} YoY% — {_SUFFIX.get(sfx, sfx)}"

    # Ret_{Ticker}{_suffix}
    if name.startswith("Ret_"):
        rest = name[4:]
        for ticker, tname in _TICKER.items():
            if rest == ticker:
                return f"{tname} 월간 수익률"
            if rest.startswith(ticker + "_"):
                sfx = rest[len(ticker) + 1:]
                return f"{tname} 월간 수익률 — {_SUFFIX.get(sfx, sfx)}"

    return "(설명 없음)"


def print_feature_descriptions(features: list):
    print(f"\n{'─'*64}")
    print(f"  최종 모델 선택 피처 ({len(features)}개)")
    print(f"{'─'*64}")
    print(f"  {'#':>3}  {'피처명':<35}  설명")
    print(f"  {'─'*3}  {'─'*35}  {'─'*30}")
    for i, feat in enumerate(features, 1):
        print(f"  {i:>3}. {feat:<35}  {_describe(feat)}")
    print(f"{'─'*64}")


# ── 메인 ───────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  Step 5  최종 평가 + 시각화")
    print("=" * 64)

    print("\n[1] 데이터 로드")
    X_tune, y_tune, X_ho, y_ho = load_data()
    print(f"  Tune: {len(X_tune)}개월  Holdout: {len(X_ho)}개월  "
          f"({X_ho.index[0].date()} ~ {X_ho.index[-1].date()})")

    print("\n[2] 최종 모델 로드")
    model, features, _ = load_model()

    print("\n[3] TimeSeriesSplit CV 평가 (5-fold)")
    fold_results = run_cv_evaluation(model, features, X_tune, y_tune)
    avg_metrics  = {k: _avg(fold_results, k) for k in METRIC_KEYS}

    print("\n[4] Hold-out 평가")
    holdout_metrics, _ = evaluate_holdout(model, features, X_tune, y_tune, X_ho, y_ho)

    print("\n[5] 결과 요약")
    print()
    print_results(fold_results, avg_metrics, holdout_metrics)

    print("\n[6] 파일 저장")
    save_csv(fold_results, avg_metrics, holdout_metrics)
    fig1_prediction_timeline(model, features, X_tune, y_tune, X_ho, y_ho)
    fig2_cv_metrics(avg_metrics, holdout_metrics)
    fig3_bear_improvement(avg_metrics)

    def _fmt(v, pct=False):
        if v is None: return "N/A"
        return f"{v:.1f}%" if pct else f"{v:.3f}"

    print("\n" + "=" * 64)
    print("  최종 결과 (CV 평균)")
    print(f"  RMSE={_fmt(avg_metrics['rmse'])}  Bull={_fmt(avg_metrics.get('rmse_bull'))}  Bear={_fmt(avg_metrics.get('rmse_bear'))}")
    print(f"  DirAcc={_fmt(avg_metrics['dir_acc'], True)}  Bull={_fmt(avg_metrics.get('dir_bull'), True)}  Bear={_fmt(avg_metrics.get('dir_bear'), True)}")
    print(f"  AsymLoss={_fmt(avg_metrics['asym_loss'])}")
    print("=" * 64)

    print_feature_descriptions(features)

    print("  Step 5 완료.")


if __name__ == "__main__":
    main()
