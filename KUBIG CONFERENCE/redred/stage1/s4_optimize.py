"""
s4_optimize.py — Step 4: Bear 최적화 최종 모델
================================================
두 가지 전략으로 Bear 구간 예측 성능을 향상시킨다:
  1. Optuna 목적함수를 RMSE → Asymmetric Loss로 교체
     (Bear 오예측에 더 높은 페널티: Bull오답=2.0, Bear오답=3.0)
  2. model.fit()에 Bear 월 sample_weight=2.0, Bull 월=1.0 적용

입력:  outputs/data/features_dataset.csv
       outputs/models/best_xgboost_selected.pkl  (Step 3 결과 — 피처 목록 참조용)
출력:  outputs/models/best_xgboost_final.pkl     (최종 배포 모델)
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import (
    FEATURES_PATH, SELECTED_PKL, FINAL_PKL,
    PRIMARY_TARGET, TEST_EVAL_SIZE,
    N_SPLITS, TEST_SIZE, MIN_TRAIN,
    N_TRIALS, RANDOM_STATE,
    W_BULL_CORRECT, W_BULL_WRONG, W_BEAR_CORRECT, W_BEAR_WRONG,
    BEAR_SAMPLE_W,
)


# ── 데이터 로드 ────────────────────────────────────────────────
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


# ── Bear 샘플 가중치 ───────────────────────────────────────────
def bear_weights(y) -> np.ndarray:
    """Bull(YoY>0)=1.0, Bear(YoY≤0)=2.0"""
    return np.where(np.asarray(y) > 0, 1.0, BEAR_SAMPLE_W)


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


# ── Asymmetric Loss CV ─────────────────────────────────────────
def cv_asymloss(params: dict, X: pd.DataFrame, y: pd.Series) -> float:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    losses = []
    for tr, te in tscv.split(X):
        if len(tr) < MIN_TRAIN: continue
        w_tr = bear_weights(y.iloc[tr])
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w_tr)
        preds = m.predict(X.iloc[te])
        losses.append(compute_all_metrics(y.iloc[te].values, preds)["asym_loss"])
    return float(np.mean(losses)) if losses else float("inf")


# ── Optuna Objective ───────────────────────────────────────────
def objective(trial, X, y):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1,
    }
    return cv_asymloss(params, X, y)


# ── Optuna 탐색 ────────────────────────────────────────────────
def run_optuna(X, y) -> dict:
    print(f"  Optuna 시작 (n_trials={N_TRIALS}, 목적: AsymLoss 최소화)")
    study = optuna.create_study(
        study_name="xgboost_bear_optimization",
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == N_TRIALS - 1:
            print(f"  Trial {trial.number+1:3d}/{N_TRIALS}  "
                  f"AsymLoss(best): {study.best_value:.4f}")

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=N_TRIALS, callbacks=[callback], show_progress_bar=False,
    )
    print(f"  → 최적 AsymLoss: {study.best_value:.4f}")
    print(f"  → 최적 파라미터: {study.best_params}")
    return study.best_params


# ── 메인 ───────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  Step 4  Bear 최적화 최종 모델")
    print("  전략: (1) AsymLoss Optuna  (2) Bear sample_weight=2.0")
    print("=" * 64)

    print("\n[1] 데이터 로드")
    X_tune, y_tune, X_ho, y_ho = load_data()
    bear_n = int((y_tune.values <= 0).sum())
    bull_n = int((y_tune.values >  0).sum())
    print(f"  Tune {len(X_tune)}개월  (Bear: {bear_n}개월 / Bull: {bull_n}개월)")

    print("\n[2] 선택 피처 로드")
    with open(SELECTED_PKL, "rb") as f:
        data = pickle.load(f)
    features = data["feature_names"]
    print(f"  선택 피처: {len(features)}개")

    print("\n[3] Optuna 하이퍼파라미터 탐색")
    best_params = run_optuna(X_tune[features], y_tune)

    print("\n[4] 최종 모델 학습 + 저장")
    params = {**best_params, "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
    w_tune = bear_weights(y_tune)
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X_tune[features], y_tune, sample_weight=w_tune)

    with open(FINAL_PKL, "wb") as f:
        pickle.dump({
            "model":         final_model,
            "feature_names": features,
            "best_params":   best_params,
        }, f)
    print(f"  → 최종 모델 저장: {FINAL_PKL}")

    # 간단한 CV 성능 확인
    print("\n[5] CV 성능 확인")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    fold_metrics = []
    X_sub = X_tune[features]
    for fold_i, (tr, te) in enumerate(tscv.split(X_sub), 1):
        if len(tr) < MIN_TRAIN: continue
        m = xgb.XGBRegressor(**params)
        m.fit(X_sub.iloc[tr], y_tune.iloc[tr], sample_weight=bear_weights(y_tune.iloc[tr]))
        fold_metrics.append(compute_all_metrics(y_tune.iloc[te].values, m.predict(X_sub.iloc[te])))

    def avg(key):
        vals = [f[key] for f in fold_metrics if f[key] is not None]
        return float(np.mean(vals)) if vals else None

    def fmt(v, pct=False):
        if v is None: return "  N/A"
        return f"{v:.1f}%" if pct else f"{v:.4f}"

    print(f"\n  [CV 평균]")
    print(f"  RMSE(전체)={fmt(avg('rmse'))}  RMSE(Bull)={fmt(avg('rmse_bull'))}  RMSE(Bear)={fmt(avg('rmse_bear'))}")
    print(f"  DirAcc(전체)={fmt(avg('dir_acc'),True)}  DirAcc(Bull)={fmt(avg('dir_bull'),True)}  DirAcc(Bear)={fmt(avg('dir_bear'),True)}")
    print(f"  AsymLoss={fmt(avg('asym_loss'))}")

    print("  Step 4 완료.")


if __name__ == "__main__":
    main()
