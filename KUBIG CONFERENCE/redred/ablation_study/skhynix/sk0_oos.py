"""
sk0_oos.py — Step 0: 공급 모델 OOS 예측값 추출
================================================
Stage 1 best_xgboost_final.pkl의 feature_names + best_params를 재사용해
expanding-window walk-forward 방식으로 leakage 없는 OOS 예측값을 생성한다.

입력:  model/outputs/data/features_dataset.csv
       model/outputs/models/best_xgboost_final.pkl
출력:  skhynix/outputs/data/wsts_oos_preds.parquet
       columns: [date, wsts_pred_t6]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    FEATURES_PATH, FINAL_PKL, PRIMARY_TARGET,
    TEST_EVAL_SIZE, BEAR_SAMPLE_W,
)
from sk_config import OOS_PRED_PATH, MIN_TRAIN_M, RANDOM_STATE


def bear_weights(y: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(y) > 0, 1.0, BEAR_SAMPLE_W)


def main():
    print("=" * 64)
    print("  Step 0  공급 모델 OOS 예측값 생성 (Expanding Walk-Forward)")
    print("=" * 64)

    print("\n[1] 데이터 로드")
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    target_col = PRIMARY_TARGET
    if target_col not in df.columns:
        target_col = [c for c in df.columns if c.startswith("TARGET_")][0]
    feature_cols = [c for c in df.columns if not c.startswith("TARGET_")]
    df_clean = df.dropna(subset=[target_col])
    X_all = df_clean[feature_cols].ffill().fillna(0)
    y_all = df_clean[target_col]
    # holdout 제외 (Step 3 평가에서 leakage 방지)
    n_cv = len(X_all) - TEST_EVAL_SIZE
    X_cv, y_cv = X_all.iloc[:n_cv], y_all.iloc[:n_cv]
    print(f"  전체: {len(X_all)}개월  CV 구간: {n_cv}개월  holdout: {TEST_EVAL_SIZE}개월")

    print("\n[2] Stage 1 모델 파라미터 로드")
    with open(FINAL_PKL, "rb") as f:
        saved = pickle.load(f)
    features = saved["feature_names"]
    best_params = saved["best_params"]
    params = {**best_params, "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
    print(f"  선택 피처: {len(features)}개")
    print(f"  n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')}")

    print(f"\n[3] Walk-Forward OOS 예측 (min_train={MIN_TRAIN_M}개월, step=1개월)")
    total = n_cv - MIN_TRAIN_M
    preds = []
    dates = []
    milestone = max(1, total // 10)

    for i in range(MIN_TRAIN_M, n_cv):
        X_tr = X_cv.iloc[:i][features]
        y_tr = y_cv.iloc[:i]
        X_te = X_cv.iloc[i:i+1][features]

        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, sample_weight=bear_weights(y_tr.values))
        preds.append(float(m.predict(X_te)[0]))
        dates.append(X_cv.index[i])

        done = i - MIN_TRAIN_M + 1
        if done % milestone == 0 or done == total:
            pct = done / total * 100
            print(f"  진행: {done:3d}/{total} ({pct:5.1f}%)  날짜: {dates[-1].strftime('%Y-%m')}")

    print(f"\n[4] 저장")
    result = pd.DataFrame({"wsts_pred_t6": preds}, index=pd.DatetimeIndex(dates))
    result.index.name = "date"
    result.to_parquet(OOS_PRED_PATH)

    print(f"  → 저장: {OOS_PRED_PATH}")
    print(f"  Shape:  {result.shape}")
    print(f"  날짜:   {result.index[0].strftime('%Y-%m')} ~ {result.index[-1].strftime('%Y-%m')}")
    print(f"  예측값: min={result['wsts_pred_t6'].min():.2f}  max={result['wsts_pred_t6'].max():.2f}  "
          f"mean={result['wsts_pred_t6'].mean():.2f}")
    print("  Step 0 완료.")


if __name__ == "__main__":
    main()
