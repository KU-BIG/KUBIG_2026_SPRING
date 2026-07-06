"""
s2_tune.py — Step 2: Optuna 하이퍼파라미터 튜닝 (RMSE 목적함수)
================================================================
XGBoost 하이퍼파라미터를 TimeSeriesSplit CV 평균 RMSE 기준으로 최적화한다.
이 결과(best_xgboost.pkl)는 Step 3 피처 선택의 초기 파라미터로 사용된다.

입력:  outputs/data/features_dataset.csv
출력:  outputs/models/best_xgboost.pkl
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
    FEATURES_PATH, TUNED_PKL,
    PRIMARY_TARGET, TEST_EVAL_SIZE,
    N_SPLITS, TEST_SIZE, MIN_TRAIN,
    N_TRIALS, RANDOM_STATE,
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
    # hold-out은 Step 4 이후를 위해 제외
    split = len(X) - TEST_EVAL_SIZE
    return X.iloc[:split], y.iloc[:split]


# ── CV RMSE (Optuna 목적함수 내부) ─────────────────────────────
def cv_rmse(params: dict, X: pd.DataFrame, y: pd.Series) -> float:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < MIN_TRAIN:
            continue
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[test_idx])
        scores.append(float(np.sqrt(mean_squared_error(y.iloc[test_idx], preds))))
    return float(np.mean(scores)) if scores else float("inf")


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
    return cv_rmse(params, X, y)


# ── Optuna 탐색 ────────────────────────────────────────────────
def run_optuna(X, y) -> dict:
    print(f"  Optuna 시작 (n_trials={N_TRIALS}, 목적: CV RMSE 최소화)")
    study = optuna.create_study(
        study_name="xgboost_rmse_tuning",
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == N_TRIALS - 1:
            print(f"  Trial {trial.number+1:3d}/{N_TRIALS}  "
                  f"CV RMSE(best): {study.best_value:.4f}")

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=N_TRIALS,
        callbacks=[callback],
        show_progress_bar=False,
    )
    print(f"  → 최적 CV RMSE: {study.best_value:.4f}")
    print(f"  → 최적 파라미터: {study.best_params}")
    return study.best_params


# ── 메인 ───────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  Step 2  Optuna 하이퍼파라미터 튜닝 (RMSE 목적함수)")
    print("=" * 64)

    print("\n[1] 데이터 로드")
    X_tune, y_tune = load_data()
    print(f"  Tune: {len(X_tune)}개월  {X_tune.shape[1]}개 피처")

    print("\n[2] Optuna 탐색")
    best_params = run_optuna(X_tune, y_tune)

    print("\n[3] 최종 모델 저장")
    params = {**best_params, "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
    model = xgb.XGBRegressor(**params)
    model.fit(X_tune, y_tune)
    with open(TUNED_PKL, "wb") as f:
        pickle.dump({
            "model":         model,
            "feature_names": list(X_tune.columns),
            "best_params":   best_params,
        }, f)
    print(f"  → 저장: {TUNED_PKL}")
    print("  Step 2 완료.")


if __name__ == "__main__":
    main()
