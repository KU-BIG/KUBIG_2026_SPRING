"""
s6_evaluate.py — Step 6: 최종 평가 + 시각화
=============================================
skh_xgb_final.pkl을 로드해 7개 표준 지표 + IC + 시뮬레이션 수익률을
TimeSeriesSplit CV 및 Hold-out으로 평가하고 4종 시각화를 생성한다.

평가 지표:
  1. RMSE (전체 / Bull / Bear)
  2. Direction Accuracy (전체 / Bull / Bear)
  3. Asymmetric Loss
  4. IC (Spearman Rank Correlation)

시각화:
  01_return_timeline.png     — 예측 vs 실제 수익률 전 기간
  02_cv_metrics.png          — 평가 지표 CV vs Hold-out 바차트
  03_direction_analysis.png  — Bull/Bear 방향 정확도 + 혼동행렬
  04_simulation.png          — Long-only 전략 vs Buy & Hold 시뮬레이션

입력:  outputs/data/stage2_features.csv
       outputs/models/skh_xgb_final.pkl
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
from scipy import stats
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

METRIC_KEYS = ["rmse", "rmse_bull", "rmse_bear",
               "dir_acc", "dir_bull", "dir_bear", "asym_loss", "ic"]
METRIC_LABELS = {
    "rmse":      "RMSE\n(전체,%)",
    "rmse_bull": "RMSE\n(Bull)",
    "rmse_bear": "RMSE\n(Bear)",
    "dir_acc":   "DirAcc\n(전체,%)",
    "dir_bull":  "DirAcc\n(Bull,%)",
    "dir_bear":  "DirAcc\n(Bear,%)",
    "asym_loss": "AsymLoss",
    "ic":        "IC\n(Spearman)",
}
DIR_METRICS = {"dir_acc", "dir_bull", "dir_bear"}


# ──────────────────────────────────────────────────────────────
# 데이터 / 모델 로드
# ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    feat_cols = [c for c in df.columns if c != PRIMARY_TARGET]
    df_clean  = df.dropna(subset=[PRIMARY_TARGET])
    X = df_clean[feat_cols].ffill().fillna(0)
    y = df_clean[PRIMARY_TARGET]
    split = len(X) - TEST_EVAL_SIZE
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def dynamic_weights(y, recency_scale: float = 0.0) -> np.ndarray:
    """Recency weight(최근 시점일수록 지수적으로 증가) x 기존 bear weight."""
    y_arr = np.asarray(y)
    n = len(y_arr)
    recency_w = np.exp(np.linspace(0, recency_scale, n))
    recency_w = recency_w / recency_w.mean()
    bear_w = np.where(y_arr > 0, 1.0, BEAR_SAMPLE_W)
    return recency_w * bear_w


def load_model():
    with open(FINAL_PKL, "rb") as f:
        data = pickle.load(f)
    use_dw    = bool(data.get("use_dynamic_weights", False))
    rec_scale = float(data.get("recency_scale", 0.0))
    print(f"  최종 모델 로드: 피처 {len(data['feature_names'])}개  "
          f"dynamic_weights={use_dw}  recency_scale={rec_scale:.4f}")
    return (data["model"], data["feature_names"],
            data.get("best_params", {}), use_dw, rec_scale)


# ──────────────────────────────────────────────────────────────
# 지표 계산
# ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bull    = y_true > 0
    bear    = ~bull
    correct = (y_true > 0) == (y_pred > 0)
    w = np.where(bull & correct,  W_BULL_CORRECT,
        np.where(bull & ~correct, W_BULL_WRONG,
        np.where(bear & correct,  W_BEAR_CORRECT, W_BEAR_WRONG)))

    def safe_rmse(m): return float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))) if m.any() else None
    def safe_dir(m):  return float(correct[m].mean() * 100) if m.any() else None

    ic_val, _ = stats.spearmanr(y_true, y_pred)

    return {
        "rmse":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmse_bull": safe_rmse(bull),
        "rmse_bear": safe_rmse(bear),
        "dir_acc":   float(correct.mean() * 100),
        "dir_bull":  safe_dir(bull),
        "dir_bear":  safe_dir(bear),
        "asym_loss": float(np.sqrt((w * (y_true - y_pred) ** 2).sum() / w.sum())),
        "ic":        float(ic_val) if not np.isnan(ic_val) else None,
    }


def avg_metrics(fold_results: list) -> dict:
    result = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in fold_results if r.get(k) is not None]
        result[k] = float(np.mean(vals)) if vals else None
    return result


# ──────────────────────────────────────────────────────────────
# CV / Hold-out 평가
# ──────────────────────────────────────────────────────────────

def run_cv(model, X_tune: pd.DataFrame, y_tune: pd.Series,
           use_dynamic_weights: bool = False,
           recency_scale: float = 0.0) -> list:
    params = model.get_params()
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    results = []
    for fold_i, (tr, te) in enumerate(tscv.split(X_tune), 1):
        if len(tr) < MIN_TRAIN:
            continue
        if use_dynamic_weights:
            w_tr = dynamic_weights(y_tune.iloc[tr], recency_scale)
        else:
            w_tr = np.where(y_tune.iloc[tr].values > 0, 1.0, BEAR_SAMPLE_W)
        m = xgb.XGBRegressor(**params)
        m.fit(X_tune.iloc[tr], y_tune.iloc[tr], sample_weight=w_tr)
        preds  = m.predict(X_tune.iloc[te])
        y_t    = y_tune.iloc[te].values
        result = compute_metrics(y_t, preds)
        result["fold"]   = f"Fold {fold_i}"
        result["period"] = (f"{y_tune.index[te[0]].strftime('%Y-%m')} ~ "
                            f"{y_tune.index[te[-1]].strftime('%Y-%m')}")
        result["te_idx"]   = te
        result["te_preds"] = preds
        results.append(result)
    return results


def run_holdout(model, X_tune: pd.DataFrame, y_tune: pd.Series,
                X_ho: pd.DataFrame, y_ho: pd.Series,
                use_dynamic_weights: bool = False,
                recency_scale: float = 0.0):
    params = model.get_params()
    if use_dynamic_weights:
        w_tune = dynamic_weights(y_tune, recency_scale)
    else:
        w_tune = np.where(y_tune.values > 0, 1.0, BEAR_SAMPLE_W)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tune, y_tune, sample_weight=w_tune)
    preds   = m.predict(X_ho)
    metrics = compute_metrics(y_ho.values, preds)
    metrics["fold"]   = "Hold-out"
    metrics["period"] = (f"{y_ho.index[0].strftime('%Y-%m')} ~ "
                         f"{y_ho.index[-1].strftime('%Y-%m')}")
    return metrics, preds, m


# ──────────────────────────────────────────────────────────────
# 결과 출력 / CSV
# ──────────────────────────────────────────────────────────────

def print_results(fold_results: list, cv_avg: dict, holdout: dict):
    def _f(v, key):
        if v is None: return "  N/A "
        if key == "ic": return f"{v:+.3f}"
        return f"{v:.1f}%" if key in DIR_METRICS else f"{v:.3f}"

    hdr = (f"  {'구간':<24} {'RMSE':>7} {'Bull':>7} {'Bear':>7} "
           f"{'DirAcc':>8} {'DBull':>7} {'DBear':>7} {'Asym':>7} {'IC':>7}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr); print(sep)
    for r in fold_results:
        print(f"  {r['fold']:<24} "
              f"{_f(r['rmse'],'rmse'):>7} {_f(r['rmse_bull'],'rmse'):>7} "
              f"{_f(r['rmse_bear'],'rmse'):>7} "
              f"{_f(r['dir_acc'],'dir_acc'):>8} {_f(r['dir_bull'],'dir_bull'):>7} "
              f"{_f(r['dir_bear'],'dir_bear'):>7} "
              f"{_f(r['asym_loss'],'asym_loss'):>7} {_f(r['ic'],'ic'):>7}")
    print(sep)
    r = cv_avg
    print(f"  {'CV 평균':<24} "
          f"{_f(r['rmse'],'rmse'):>7} {_f(r['rmse_bull'],'rmse'):>7} "
          f"{_f(r['rmse_bear'],'rmse'):>7} "
          f"{_f(r['dir_acc'],'dir_acc'):>8} {_f(r['dir_bull'],'dir_bull'):>7} "
          f"{_f(r['dir_bear'],'dir_bear'):>7} "
          f"{_f(r['asym_loss'],'asym_loss'):>7} {_f(r['ic'],'ic'):>7}")
    print(sep)
    r = holdout
    print(f"  {'Hold-out (OOS)':<24} "
          f"{_f(r['rmse'],'rmse'):>7} {_f(r['rmse_bull'],'rmse'):>7} "
          f"{_f(r['rmse_bear'],'rmse'):>7} "
          f"{_f(r['dir_acc'],'dir_acc'):>8} {_f(r['dir_bull'],'dir_bull'):>7} "
          f"{_f(r['dir_bear'],'dir_bear'):>7} "
          f"{_f(r['asym_loss'],'asym_loss'):>7} {_f(r['ic'],'ic'):>7}")


def save_csv(fold_results: list, cv_avg: dict, holdout: dict):
    cols = ["fold", "period"] + METRIC_KEYS
    rows = [{k: r.get(k) for k in cols} for r in fold_results]
    rows.append({**{k: cv_avg.get(k) for k in METRIC_KEYS},
                 "fold": "CV 평균", "period": "-"})
    rows.append({**{k: holdout.get(k) for k in METRIC_KEYS},
                 "fold": "Hold-out", "period": holdout.get("period", "-")})
    path = os.path.join(METRIC_DIR, "final_cv_metrics.csv")
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → CSV 저장: {path}")


# ──────────────────────────────────────────────────────────────
# Figure 1: 예측 vs 실제 수익률 타임라인
# ──────────────────────────────────────────────────────────────

def fig1_timeline(model, X_tune, y_tune, X_ho, y_ho, holdout_preds,
                  use_dynamic_weights: bool = False,
                  recency_scale: float = 0.0):
    params = model.get_params()
    if use_dynamic_weights:
        w_tune = dynamic_weights(y_tune, recency_scale)
    else:
        w_tune = np.where(y_tune.values > 0, 1.0, BEAR_SAMPLE_W)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tune, y_tune, sample_weight=w_tune)
    tune_preds = m.predict(X_tune)

    y_all    = pd.concat([y_tune, y_ho])
    p_all    = np.concatenate([tune_preds, holdout_preds])
    ho_start = y_ho.index[0]

    correct_ho = (y_ho.values > 0) == (holdout_preds > 0)
    dir_acc    = float(correct_ho.mean() * 100)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2, 1]})

    # ① 전체 타임라인
    ax = axes[0]
    ax.plot(y_all.index, y_all.values, label="실제 수익률",
            color="steelblue", linewidth=2.0)
    ax.plot(y_all.index, p_all, label="예측 수익률",
            color="darkorange", linewidth=1.6, linestyle="--", alpha=0.9)
    ax.axvline(ho_start, color="crimson", linewidth=1.8, linestyle=":",
               label=f"Hold-out 시작 ({ho_start.strftime('%Y-%m')})")
    ax.axvspan(ho_start, y_all.index[-1], alpha=0.07, color="crimson")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.text(0.01, 0.03,
            f"Hold-out DirAcc={dir_acc:.1f}%  (Bull={correct_ho[y_ho.values > 0].mean()*100:.1f}%"
            f"  Bear={correct_ho[y_ho.values <= 0].mean()*100:.1f}%)",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88))
    ax.set_title("SK하이닉스 6개월 종가 수익률 예측 vs 실제  (Stage 2 최종 모델)", fontsize=13)
    ax.set_ylabel("6개월 수익률 (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # ② Hold-out 확대
    ax2 = axes[1]
    ax2.plot(y_ho.index, y_ho.values, label="실제",
             color="steelblue", linewidth=2.2, marker="o", markersize=7)
    ax2.plot(y_ho.index, holdout_preds, label="예측",
             color="darkorange", linewidth=1.8, linestyle="--", marker="^", markersize=7)
    ax2.fill_between(y_ho.index, y_ho.values, holdout_preds,
                     alpha=0.13, color="crimson", label="예측 오차")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
    for d, a, p in zip(y_ho.index, y_ho.values, holdout_preds):
        ax2.axvline(d, color="green" if (a > 0) == (p > 0) else "red",
                    linewidth=0.5, alpha=0.3)
    ho_rmse = float(np.sqrt(mean_squared_error(y_ho.values, holdout_preds)))
    ax2.set_title(f"Hold-out 확대 — RMSE={ho_rmse:.2f}%  DirAcc={dir_acc:.1f}%", fontsize=11)
    ax2.set_ylabel("6개월 수익률 (%)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout(h_pad=3)
    path = os.path.join(FIG_DIR, "01_return_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure 1 저장: {path}")


# ──────────────────────────────────────────────────────────────
# Figure 2: 평가 지표 바차트
# ──────────────────────────────────────────────────────────────

def fig2_cv_metrics(cv_avg: dict, holdout: dict):
    keys   = [k for k in METRIC_KEYS if k != "ic"]   # IC는 별도 표시
    labels = [METRIC_LABELS[k] for k in keys]
    cv_v   = [cv_avg.get(k) or 0 for k in keys]
    ho_v   = [holdout.get(k)  or 0 for k in keys]

    x, w = np.arange(len(keys)), 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    b1 = ax.bar(x - w / 2, cv_v, w, label="CV 평균 (5-fold)", color="#3498db", alpha=0.85)
    b2 = ax.bar(x + w / 2, ho_v, w, label="Hold-out (OOS)",   color="#e67e22", alpha=0.85)

    def _lbl(v, key): return f"{v:.1f}%" if key in DIR_METRICS else f"{v:.3f}"
    for bar, v, k in zip(b1, cv_v, keys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                _lbl(v, k), ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color="#2980b9")
    for bar, v, k in zip(b2, ho_v, keys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                _lbl(v, k), ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color="#d35400")

    ic_cv = cv_avg.get("ic")
    ic_ho = holdout.get("ic")
    if ic_cv is not None or ic_ho is not None:
        ax.text(0.99, 0.97,
                f"IC  CV={ic_cv:+.3f}  Hold-out={ic_ho:+.3f}" if (ic_cv and ic_ho)
                else f"IC  Hold-out={ic_ho:+.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("값 (RMSE·AsymLoss: 낮을수록 / DirAcc: 높을수록 우수)")
    ax.set_title("Stage 2 평가 지표: CV 평균 vs Hold-out  (SK하이닉스 6M 수익률 예측)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "02_cv_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure 2 저장: {path}")


# ──────────────────────────────────────────────────────────────
# Figure 3: Bull/Bear 방향성 분석
# ──────────────────────────────────────────────────────────────

def fig3_direction(model, X_tune, y_tune, X_ho, y_ho, holdout_preds,
                   use_dynamic_weights: bool = False,
                   recency_scale: float = 0.0):
    params  = model.get_params()
    tscv    = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    cv_bull, cv_bear = [], []
    for tr, te in tscv.split(X_tune):
        if len(tr) < MIN_TRAIN: continue
        if use_dynamic_weights:
            w_tr = dynamic_weights(y_tune.iloc[tr], recency_scale)
        else:
            w_tr = np.where(y_tune.iloc[tr].values > 0, 1.0, BEAR_SAMPLE_W)
        m = xgb.XGBRegressor(**params)
        m.fit(X_tune.iloc[tr], y_tune.iloc[tr], sample_weight=w_tr)
        p  = m.predict(X_tune.iloc[te])
        yt = y_tune.iloc[te].values
        c  = (yt > 0) == (p > 0)
        bm = yt > 0
        cv_bull.append(float(c[bm].mean() * 100) if bm.any() else None)
        cv_bear.append(float(c[~bm].mean() * 100) if (~bm).any() else None)

    def nm(lst): return float(np.mean([v for v in lst if v is not None])) if any(v is not None for v in lst) else 0

    correct_ho = (y_ho.values > 0) == (holdout_preds > 0)
    bull_ho    = y_ho.values > 0
    ho_bull    = float(correct_ho[bull_ho].mean() * 100) if bull_ho.any() else 0
    ho_bear    = float(correct_ho[~bull_ho].mean() * 100) if (~bull_ho).any() else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ① Bull/Bear 정확도 비교
    ax = axes[0]
    x, w = np.arange(2), 0.32
    b1 = ax.bar(x - w / 2, [nm(cv_bull), ho_bull], w, label="Bull", color="#2ecc71", alpha=0.85)
    b2 = ax.bar(x + w / 2, [nm(cv_bear), ho_bear], w, label="Bear", color="#e74c3c", alpha=0.85)
    for bar, v in list(zip(b1, [nm(cv_bull), ho_bull])) + list(zip(b2, [nm(cv_bear), ho_bear])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(50, color="gray", linewidth=1.0, linestyle="--", alpha=0.6, label="50% (랜덤)")
    ax.set_xticks(x)
    ax.set_xticklabels(["CV 평균 (in-sample)", "Hold-out (OOS)"], fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title("Bull / Bear 구간별 방향 정확도", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # ② 혼동행렬 (Hold-out)
    ax2 = axes[1]
    tp = int(( (y_ho.values > 0) & (holdout_preds > 0)).sum())
    fp = int(( (y_ho.values <= 0) & (holdout_preds > 0)).sum())
    fn = int(( (y_ho.values > 0) & (holdout_preds <= 0)).sum())
    tn = int(( (y_ho.values <= 0) & (holdout_preds <= 0)).sum())
    cm = np.array([[tp, fn], [fp, tn]])
    im = ax2.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["예측 상승", "예측 하락"], fontsize=10)
    ax2.set_yticklabels(["실제 상승", "실제 하락"], fontsize=10)
    ax2.set_title("Hold-out 혼동행렬  (Bull=상승, Bear=하락)", fontsize=12)
    plt.colorbar(im, ax=ax2)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "03_direction_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure 3 저장: {path}")


# ──────────────────────────────────────────────────────────────
# Figure 4: Long-only 전략 시뮬레이션
# ──────────────────────────────────────────────────────────────

def fig4_simulation(y_ho: pd.Series, holdout_preds: np.ndarray):
    """
    예측이 양수일 때 SK하이닉스 매수 (Long-only) vs Buy & Hold 비교.
    각 분기별 수익률 합산 (단순 누적 수익률, 복리 아님).
    """
    strategy_ret = np.where(holdout_preds > 0, y_ho.values, 0.0)
    buyhold_ret  = y_ho.values

    cum_strategy = np.cumsum(strategy_ret)
    cum_buyhold  = np.cumsum(buyhold_ret)

    n_trades    = int((holdout_preds > 0).sum())
    total_ret_s = float(cum_strategy[-1])
    total_ret_b = float(cum_buyhold[-1])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_ho.index, cum_strategy, label=f"Long-only 전략  (진입: {n_trades}회)",
            color="green", linewidth=2.2)
    ax.plot(y_ho.index, cum_buyhold,  label="Buy & Hold",
            color="steelblue", linewidth=2.0, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.fill_between(y_ho.index, cum_strategy, cum_buyhold,
                    where=(cum_strategy > cum_buyhold),
                    alpha=0.15, color="green", label="전략 우위")
    ax.fill_between(y_ho.index, cum_strategy, cum_buyhold,
                    where=(cum_strategy < cum_buyhold),
                    alpha=0.15, color="red", label="전략 열위")

    ax.text(0.02, 0.95,
            f"전략 누적: {total_ret_s:+.1f}%   B&H 누적: {total_ret_b:+.1f}%   "
            f"초과수익: {total_ret_s - total_ret_b:+.1f}%",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88))

    ax.set_ylabel("누적 수익률 (%, 단순합산)")
    ax.set_title("Hold-out 구간 Long-only 시뮬레이션 vs Buy & Hold", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "04_simulation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure 4 저장: {path}")


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Step 6  최종 평가 + 시각화")
    print("=" * 64)

    print("\n[1] 데이터 로드")
    X_tune, y_tune, X_ho, y_ho = load_data()
    print(f"  Tune: {len(X_tune)}분기  Holdout: {len(X_ho)}분기  "
          f"({X_ho.index[0].date()} ~ {X_ho.index[-1].date()})")

    print("\n[2] 최종 모델 로드")
    model, features, _, use_dw, rec_scale = load_model()

    print("\n[3] TimeSeriesSplit CV 평가 (5-fold)")
    fold_results = run_cv(model, X_tune, y_tune,
                          use_dynamic_weights=use_dw, recency_scale=rec_scale)
    cv_avg       = avg_metrics(fold_results)

    print("\n[4] Hold-out 평가")
    holdout, ho_preds, trained_model = run_holdout(
        model, X_tune, y_tune, X_ho, y_ho,
        use_dynamic_weights=use_dw, recency_scale=rec_scale)

    print("\n[5] 결과 요약")
    print()
    print_results(fold_results, cv_avg, holdout)

    print("\n[6] 파일 저장")
    save_csv(fold_results, cv_avg, holdout)
    fig1_timeline(model, X_tune, y_tune, X_ho, y_ho, ho_preds,
                  use_dynamic_weights=use_dw, recency_scale=rec_scale)
    fig2_cv_metrics(cv_avg, holdout)
    fig3_direction(model, X_tune, y_tune, X_ho, y_ho, ho_preds,
                   use_dynamic_weights=use_dw, recency_scale=rec_scale)
    fig4_simulation(y_ho, ho_preds)

    def _fmt(v, pct=False):
        if v is None: return "N/A"
        return f"{v:.1f}%" if pct else f"{v:.3f}"

    print("\n" + "=" * 64)
    print("  최종 결과 요약 (CV 평균)")
    print(f"  RMSE={_fmt(cv_avg['rmse'])}  "
          f"Bull={_fmt(cv_avg.get('rmse_bull'))}  "
          f"Bear={_fmt(cv_avg.get('rmse_bear'))}")
    print(f"  DirAcc={_fmt(cv_avg['dir_acc'], True)}  "
          f"Bull={_fmt(cv_avg.get('dir_bull'), True)}  "
          f"Bear={_fmt(cv_avg.get('dir_bear'), True)}")
    print(f"  AsymLoss={_fmt(cv_avg['asym_loss'])}  "
          f"IC={_fmt(cv_avg.get('ic'))}")
    print("=" * 64)
    print("  Step 6 완료.")


if __name__ == "__main__":
    main()
