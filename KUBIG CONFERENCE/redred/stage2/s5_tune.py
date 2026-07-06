"""
s5_tune.py — Step 5: XGBoost Optuna 튜닝
==========================================
2단계 최적화:
  Phase A: RMSE 목적함수로 초기 탐색 → skh_xgb_tuned.pkl
  Phase B: Asymmetric Loss (Bear 하방 오예측 페널티) → skh_xgb_final.pkl

CV 구조: TimeSeriesSplit (분기 단위)
  N_SPLITS=5, TEST_SIZE=4분기(1년), MIN_TRAIN=20분기(5년)

입력:  outputs/data/stage2_features.csv
출력:  outputs/models/skh_xgb_tuned.pkl
       outputs/models/skh_xgb_final.pkl
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import (
    FEATURES_PATH, TUNED_PKL, FINAL_PKL,
    PRIMARY_TARGET, TEST_EVAL_SIZE,
    N_SPLITS, TEST_SIZE, MIN_TRAIN,
    N_TRIALS, RANDOM_STATE,
    W_BULL_CORRECT, W_BULL_WRONG, W_BEAR_CORRECT, W_BEAR_WRONG,
    BEAR_SAMPLE_W,
)

# ── Dynamic Weight 설정 ───────────────────────────────────────
USE_DYNAMIC_WEIGHTS = True   # False → 기존 bear_weights() 고정 사용


def load_data():
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    feat_cols = [c for c in df.columns if c != PRIMARY_TARGET]
    df_clean  = df.dropna(subset=[PRIMARY_TARGET])
    X = df_clean[feat_cols].ffill().fillna(0)
    y = df_clean[PRIMARY_TARGET]
    split = len(X) - TEST_EVAL_SIZE
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def bear_weights(y) -> np.ndarray:
    return np.where(np.asarray(y) > 0, 1.0, BEAR_SAMPLE_W)


def dynamic_weights(y, recency_scale: float = 0.0) -> np.ndarray:
    """Recency weight(최근 시점일수록 지수적으로 증가) x 기존 bear weight."""
    y_arr = np.asarray(y)
    n = len(y_arr)
    recency_w = np.exp(np.linspace(0, recency_scale, n))
    recency_w = recency_w / recency_w.mean()
    bear_w = np.where(y_arr > 0, 1.0, BEAR_SAMPLE_W)
    return recency_w * bear_w


def asym_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    bull    = y_true > 0
    bear    = ~bull
    correct = (y_true > 0) == (y_pred > 0)
    w = np.where(bull & correct,  W_BULL_CORRECT,
        np.where(bull & ~correct, W_BULL_WRONG,
        np.where(bear & correct,  W_BEAR_CORRECT, W_BEAR_WRONG)))
    return float(np.sqrt((w * (y_true - y_pred) ** 2).sum() / w.sum()))


def cv_rmse(params: dict, X: pd.DataFrame, y: pd.Series) -> float:
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    scores = []
    for tr, te in tscv.split(X):
        if len(tr) < MIN_TRAIN:
            continue
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        preds = m.predict(X.iloc[te])
        scores.append(np.sqrt(mean_squared_error(y.iloc[te].values, preds)))
    return float(np.mean(scores)) if scores else float("inf")


def cv_asymloss(params: dict, X: pd.DataFrame, y: pd.Series,
                recency_scale: float = 0.0) -> float:
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    scores = []
    for tr, te in tscv.split(X):
        if len(tr) < MIN_TRAIN:
            continue
        if USE_DYNAMIC_WEIGHTS:
            w_tr = dynamic_weights(y.iloc[tr], recency_scale)
        else:
            w_tr = bear_weights(y.iloc[tr])
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w_tr)
        preds = m.predict(X.iloc[te])
        scores.append(asym_loss(y.iloc[te].values, preds))
    return float(np.mean(scores)) if scores else float("inf")


def build_params(trial) -> dict:
    """Optuna trial → XGBoost 파라미터 딕셔너리."""
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 15),
        "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1,
    }


def run_optuna(obj_fn, X: pd.DataFrame, y: pd.Series, study_name: str) -> dict:
    study = optuna.create_study(
        study_name=study_name, direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )

    def cb(study, trial):
        if trial.number % 10 == 0 or trial.number == N_TRIALS - 1:
            print(f"    Trial {trial.number + 1:3d}/{N_TRIALS}  "
                  f"best={study.best_value:.4f}")

    study.optimize(
        lambda t: obj_fn({**build_params(t),
                          "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1},
                         X, y),
        n_trials=N_TRIALS, callbacks=[cb], show_progress_bar=False,
    )
    print(f"  → 최적값: {study.best_value:.4f}")
    return study.best_params


def main():
    print("=" * 64)
    print("  Step 5  XGBoost Optuna 튜닝")
    print("=" * 64)

    print("\n[1] 데이터 로드")
    X_tune, y_tune, X_ho, y_ho = load_data()
    bear_n = int((y_tune.values <= 0).sum())
    bull_n = int((y_tune.values >  0).sum())
    print(f"  Tune: {len(X_tune)}분기  (Bear: {bear_n} / Bull: {bull_n})")
    print(f"  Hold-out: {len(X_ho)}분기  "
          f"({X_ho.index[0].date()} ~ {X_ho.index[-1].date()})")
    print(f"  피처: {X_tune.shape[1]}개")

    # ── Phase A: RMSE ──────────────────────────────────────────
    print("\n[2] Phase A — RMSE 최적화 (n_trials={})".format(N_TRIALS))
    best_a = run_optuna(cv_rmse, X_tune, y_tune, "skh_rmse")
    params_a = {**best_a, "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
    model_a  = xgb.XGBRegressor(**params_a)
    model_a.fit(X_tune, y_tune)

    with open(TUNED_PKL, "wb") as f:
        pickle.dump({
            "model":         model_a,
            "feature_names": list(X_tune.columns),
            "best_params":   best_a,
        }, f)
    print(f"  → 저장: {TUNED_PKL}")

    # ── Phase B: Asymmetric Loss ───────────────────────────────
    print("\n[3] Phase B — Asymmetric Loss 최적화 (Bear 페널티 W={})".format(W_BEAR_WRONG))
    if USE_DYNAMIC_WEIGHTS:
        print("  Dynamic weights 활성화 (recency_scale 공동 최적화)")

    def objective_b(trial):
        xgb_params = {**build_params(trial),
                      "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
        rs = trial.suggest_float("recency_scale", 0.0, 0.5) if USE_DYNAMIC_WEIGHTS else 0.0
        return cv_asymloss(xgb_params, X_tune, y_tune, recency_scale=rs)

    study_b = optuna.create_study(
        study_name="skh_asym", direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )

    def cb_b(study, trial):
        if trial.number % 10 == 0 or trial.number == N_TRIALS - 1:
            print(f"    Trial {trial.number + 1:3d}/{N_TRIALS}  "
                  f"best={study.best_value:.4f}")

    study_b.optimize(objective_b, n_trials=N_TRIALS, callbacks=[cb_b],
                     show_progress_bar=False)
    print(f"  → 최적값: {study_b.best_value:.4f}")
    best_b = study_b.best_params

    # recency_scale을 XGBoost 파라미터 딕셔너리에서 분리
    best_recency_scale = float(best_b.pop("recency_scale", 0.0))
    params_b = {**best_b, "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}

    if USE_DYNAMIC_WEIGHTS:
        w_tune = dynamic_weights(y_tune, best_recency_scale)
        print(f"  → 최적 recency_scale: {best_recency_scale:.4f}")
    else:
        w_tune = bear_weights(y_tune)

    model_b = xgb.XGBRegressor(**params_b)
    model_b.fit(X_tune, y_tune, sample_weight=w_tune)

    with open(FINAL_PKL, "wb") as f:
        pickle.dump({
            "model":               model_b,
            "feature_names":       list(X_tune.columns),
            "best_params":         best_b,
            "recency_scale":       best_recency_scale,
            "use_dynamic_weights": USE_DYNAMIC_WEIGHTS,
        }, f)
    print(f"  → 저장: {FINAL_PKL}")

    # ── 간단 CV 확인 ───────────────────────────────────────────
    print("\n[4] Phase B CV 성능 확인")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    fold_rmse, fold_dir = [], []
    for tr, te in tscv.split(X_tune):
        if len(tr) < MIN_TRAIN:
            continue
        if USE_DYNAMIC_WEIGHTS:
            w_cv = dynamic_weights(y_tune.iloc[tr], best_recency_scale)
        else:
            w_cv = bear_weights(y_tune.iloc[tr])
        m = xgb.XGBRegressor(**params_b)
        m.fit(X_tune.iloc[tr], y_tune.iloc[tr], sample_weight=w_cv)
        preds  = m.predict(X_tune.iloc[te])
        y_t    = y_tune.iloc[te].values
        fold_rmse.append(np.sqrt(mean_squared_error(y_t, preds)))
        fold_dir.append(float(((y_t > 0) == (preds > 0)).mean() * 100))

    print(f"  CV RMSE:   {np.mean(fold_rmse):.3f}%")
    print(f"  CV DirAcc: {np.mean(fold_dir):.1f}%")
    print("  Step 5 완료.")


if __name__ == "__main__":
    main()
