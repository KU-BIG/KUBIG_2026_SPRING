"""
s3_stage1_feat.py — Step 3: Stage 1 Expanding Window Pseudo 예측
=================================================================
Stage 1 XGBoost 모델을 각 관찰일 기준 expanding window로 재훈련하여
"관찰일 시점 6개월 선행 WSTS WW YoY% 예측값 (v2_pred_ww_yoy)"을 생성한다.

전략:
  FOR each obs_date:
    1. Stage 1 features_dataset.csv를 obs_date 이전 월까지 슬라이스
    2. Stage 1 최적 하이퍼파라미터로 expanding window 재훈련
    3. obs_date에 해당하는 행 예측 → v2_pred_ww_yoy
  → 전체 구간에서 lookahead 없는 OOS pseudo-prediction 생성

입력:  stage1/outputs/data/features_dataset.csv
       stage1/outputs/models/best_xgboost_final.pkl
       outputs/data/quarterly_dates.csv
       outputs/data/raw_quarterly.csv    (현재 WSTS YoY 참조용)
출력:  outputs/data/stage1_predictions.csv
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import (
    DATES_PATH, S1PRED_PATH, RAW_PATH,
    STAGE1_FEATURES_PATH, STAGE1_FINAL_PKL,
    RANDOM_STATE,
)

STAGE1_TARGET    = "TARGET_Worldwide_YoY_T6"
STAGE1_MIN_TRAIN = 60   # 최소 60개월 필요 (5년)


def load_stage1_assets():
    """Stage 1 피처 데이터셋 + 하이퍼파라미터 + 피처 목록 로드."""
    df = pd.read_csv(STAGE1_FEATURES_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)

    with open(STAGE1_FINAL_PKL, "rb") as f:
        s1 = pickle.load(f)

    features    = s1["feature_names"]
    best_params = s1["best_params"]
    params = {**best_params, "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1}
    return df, features, params


def predict_at_obs(df_s1: pd.DataFrame, features: list, params: dict,
                   obs_date: pd.Timestamp) -> float:
    """
    obs_date 기준 expanding window로 Stage 1 재훈련 후 예측.

    - train: obs_date 이전 월까지의 데이터 (타겟 NaN 행 제외)
    - predict: obs_date가 포함된 월의 행
    """
    # obs_date → 해당 월의 월말 (Stage 1 인덱스 기준)
    snap = (obs_date + pd.offsets.MonthEnd(0))

    df_train = df_s1[df_s1.index < snap].dropna(subset=[STAGE1_TARGET])
    if len(df_train) < STAGE1_MIN_TRAIN:
        return np.nan

    df_snap = df_s1[df_s1.index <= snap]
    if df_snap.empty:
        return np.nan

    avail = [f for f in features if f in df_train.columns]
    if not avail:
        return np.nan

    X_train = df_train[avail].ffill().fillna(0)
    y_train = df_train[STAGE1_TARGET]
    X_pred  = df_snap[avail].ffill().fillna(0).iloc[[-1]]

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return float(model.predict(X_pred)[0])


def main():
    print("=" * 64)
    print("  Step 3  Stage 1 Expanding Window Pseudo 예측")
    print("=" * 64)

    print("\n[1] Stage 1 데이터 + 모델 로드")
    df_s1, features, params = load_stage1_assets()
    print(f"  Stage 1 피처: {len(features)}개  |  데이터: {len(df_s1)}개월  "
          f"({df_s1.index.min().date()} ~ {df_s1.index.max().date()})")
    print(f"  하이퍼파라미터: n_estimators={params.get('n_estimators')}, "
          f"max_depth={params.get('max_depth')}, "
          f"learning_rate={params.get('learning_rate'):.4f}")

    print("\n[2] 관찰일 로드")
    df_dates  = pd.read_csv(DATES_PATH, parse_dates=["obs_date"])
    obs_dates = sorted(df_dates["obs_date"].dropna().unique())
    print(f"  관찰일 {len(obs_dates)}개")

    print("\n[3] Expanding Window 예측 생성 중...")
    results = []
    for obs_d in tqdm(obs_dates, desc="  Stage 1 재훈련"):
        pred = predict_at_obs(df_s1, features, params, pd.Timestamp(obs_d))
        results.append({"obs_date": obs_d, "v2_pred_ww_yoy": pred})

    df_pred = pd.DataFrame(results).set_index("obs_date")
    df_pred.index = pd.to_datetime(df_pred.index)

    # v2_pred_vs_current: Stage 1 예측값 - 관찰일 현재 WSTS WW YoY%
    try:
        df_raw = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
        if "WSTS_WW_YoY" in df_raw.columns:
            df_pred["v2_pred_vs_current"] = (
                df_pred["v2_pred_ww_yoy"] - df_raw["WSTS_WW_YoY"]
            )
    except Exception:
        pass

    valid   = df_pred["v2_pred_ww_yoy"].notna().sum()
    skipped = len(obs_dates) - valid
    print(f"\n  예측 생성: {valid}/{len(obs_dates)}개  "
          f"(건너뜀: {skipped}개 — 학습 데이터 부족)")

    valid_preds = df_pred["v2_pred_ww_yoy"].dropna()
    print(f"  예측값 범위: {valid_preds.min():.1f}% ~ {valid_preds.max():.1f}%")
    print(f"  예측값 평균: {valid_preds.mean():.1f}%  표준편차: {valid_preds.std():.1f}%")

    df_pred.to_csv(S1PRED_PATH)
    print(f"\n  → 저장: {S1PRED_PATH}")
    print("  Step 3 완료.")


if __name__ == "__main__":
    main()
