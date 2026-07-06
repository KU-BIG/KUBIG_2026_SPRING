"""
s4_features.py — Step 4: 피처 엔지니어링 + 최종 데이터셋 구성
==============================================================
raw_quarterly + stage1_predictions + quarterly_dates를 병합하고
달력·사이클 피처를 추가한 뒤 피처 선택을 수행한다.

피처 선택 3단계:
  1. NaN 비율 50% 초과 제거
  2. VIF 기반 다중공선성 제거 (임계값 10)
  3. XGBoost importance 상위 60% → RFE 최종 25개

입력:  outputs/data/quarterly_dates.csv
       outputs/data/raw_quarterly.csv
       outputs/data/stage1_predictions.csv
출력:  outputs/data/stage2_features.csv
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import xgboost as xgb

warnings.filterwarnings("ignore")

from config import (
    DATES_PATH, RAW_PATH, S1PRED_PATH, FEATURES_PATH,
    PRIMARY_TARGET, RANDOM_STATE,
)

VIF_THRESHOLD      = 10.0
IMPORTANCE_TOP_PCT = 0.60
RFE_N_FEATURES     = 25

# 피처 선택에서 항상 유지할 핵심 피처 (제거 안 함)
PROTECTED = [
    "SKH_ret_6m", "SKH_ret_12m", "SKH_price_obs",
    "WSTS_WW_YoY", "v2_pred_ww_yoy",
]


# ──────────────────────────────────────────────────────────────
# 달력·사이클 피처
# ──────────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index)

    # 관찰월 → 실적 발표 분기 매핑
    # 관찰월: 1→Q2실적 / 4→Q3 / 7→Q4 / 10→Q1
    obs_month = idx.month
    df["earnings_quarter"] = pd.Series(
        obs_month, index=df.index
    ).map({1: 2, 4: 3, 7: 4, 10: 1}).values

    df["quarter_sin"] = np.sin(2 * np.pi * df["earnings_quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["earnings_quarter"] / 4)

    # 4년 반도체 슈퍼사이클 위치 (0~1)
    year = idx.year
    df["supercycle_pos"] = ((year - 2000) % 4) / 4.0

    # 장기 추세 proxy
    df["years_since_2000"] = (year - 2000).astype(float)

    return df


# ──────────────────────────────────────────────────────────────
# 피처 선택 함수
# ──────────────────────────────────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.Series:
    """sklearn LinearRegression으로 VIF 계산 (statsmodels 불필요)."""
    vifs = {}
    X_vals = X.values.astype(float)
    for i, col in enumerate(X.columns):
        y_i = X_vals[:, i]
        X_others = np.delete(X_vals, i, axis=1)
        if X_others.shape[1] == 0:
            vifs[col] = 1.0
            continue
        lr = LinearRegression(fit_intercept=True).fit(X_others, y_i)
        ss_res = np.sum((y_i - lr.predict(X_others)) ** 2)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        r2 = min(max(r2, 0), 0.9999)
        vifs[col] = 1 / (1 - r2)
    return pd.Series(vifs)


def remove_vif(X: pd.DataFrame, threshold: float, protected: list) -> list:
    cols = list(X.columns)
    while True:
        X_sub = X[cols].values.astype(float)
        vifs  = compute_vif(pd.DataFrame(X_sub, columns=cols))
        removable = vifs[~vifs.index.isin(protected)]
        if removable.empty or removable.max() <= threshold:
            break
        worst = removable.idxmax()
        print(f"    VIF 제거: {worst} (VIF={vifs[worst]:.1f})")
        cols.remove(worst)
    return cols


def select_by_importance(X: pd.DataFrame, y: pd.Series,
                         top_pct: float, protected: list) -> list:
    m = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                          random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    m.fit(X.values, y.values)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    n_keep = max(int(len(imp) * top_pct), RFE_N_FEATURES + 5)
    top = list(imp.head(n_keep).index)
    # 보호 피처가 빠지면 강제 추가
    for p in protected:
        if p in X.columns and p not in top:
            top.append(p)
    return top


def select_by_rfe(X: pd.DataFrame, y: pd.Series,
                  n_features: int, protected: list) -> list:
    # RFE 대상: 보호 피처 외 나머지
    non_protected = [c for c in X.columns if c not in protected]
    n_select = max(n_features - len([p for p in protected if p in X.columns]), 1)

    if len(non_protected) <= n_select:
        return list(X.columns)

    base = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                             random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    rfe = RFE(base, n_features_to_select=n_select, step=2)
    rfe.fit(X[non_protected].values, y.values)
    selected_non_protected = list(np.array(non_protected)[rfe.support_])

    result = selected_non_protected + [p for p in protected if p in X.columns]
    return list(dict.fromkeys(result))   # 순서 유지 + 중복 제거


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Step 4  피처 엔지니어링 + 최종 데이터셋 구성")
    print("=" * 64)

    # ── 데이터 로드 + 병합 ──────────────────────────────────────
    print("\n[1] 데이터 로드 + 병합")
    df_dates = pd.read_csv(DATES_PATH, parse_dates=["obs_date", "earnings_date"])
    df_dates = df_dates.set_index("obs_date")
    df_dates.index = pd.to_datetime(df_dates.index)

    df_raw  = pd.read_csv(RAW_PATH,    index_col=0, parse_dates=True)
    df_s1   = pd.read_csv(S1PRED_PATH, index_col=0, parse_dates=True)

    df = df_raw.join(df_s1, how="left")
    df = df.join(df_dates[[PRIMARY_TARGET]], how="left")
    print(f"  병합 후: {len(df)}개 관찰 × {len(df.columns)}개 컬럼")

    # ── 달력·사이클 피처 추가 ────────────────────────────────────
    print("\n[2] 달력·사이클 피처 추가")
    df = add_calendar_features(df)

    # ── v2 파생 피처 ────────────────────────────────────────────
    if "v2_pred_ww_yoy" in df.columns and "WSTS_WW_YoY" in df.columns:
        df["v2_pred_vs_current"] = df["v2_pred_ww_yoy"] - df["WSTS_WW_YoY"]
        df["v2_pred_bull"]       = (df["v2_pred_ww_yoy"] > 0).astype(float)
        df["v2_pred_bull"].where(df["v2_pred_ww_yoy"].notna(), np.nan, inplace=True)

    # ── 피처 / 타겟 분리 ─────────────────────────────────────────
    excl = {"price_obs", "price_earnings", "earnings_date", PRIMARY_TARGET}
    feat_cols = [c for c in df.columns if c not in excl and not c.startswith("TARGET_")]

    df_clean = df.dropna(subset=[PRIMARY_TARGET])
    X = df_clean[feat_cols].ffill()
    y = df_clean[PRIMARY_TARGET]

    print(f"\n  타겟 유효 샘플: {len(df_clean)}개")
    print(f"  타겟 범위: {y.min():.1f}% ~ {y.max():.1f}%  "
          f"평균: {y.mean():.1f}%  표준편차: {y.std():.1f}%")

    # ── 피처 선택 ────────────────────────────────────────────────
    print("\n[3] 피처 선택")

    # 3-0. NaN 50% 초과 제거
    valid_cols = [c for c in X.columns if X[c].notna().mean() >= 0.5]
    X = X[valid_cols].fillna(X[valid_cols].median())
    print(f"  3-0. NaN 필터 후: {len(valid_cols)}개")

    # 보호 피처 중 실제 존재하는 것만
    protected = [p for p in PROTECTED if p in X.columns]

    # 3-1. VIF 다중공선성 제거
    print("  3-1. VIF 다중공선성 제거")
    selected = remove_vif(X, VIF_THRESHOLD, protected)
    print(f"    → {len(selected)}개 남음")

    # 3-2. XGBoost importance 상위 60%
    print("  3-2. XGBoost importance 상위 60%")
    selected = select_by_importance(X[selected], y, IMPORTANCE_TOP_PCT, protected)
    print(f"    → {len(selected)}개 남음")

    # 3-3. RFE 최종 선별
    print(f"  3-3. RFE 최종 {RFE_N_FEATURES}개 선택")
    if len(selected) > RFE_N_FEATURES:
        selected = select_by_rfe(X[selected], y, RFE_N_FEATURES, protected)
    print(f"    → {len(selected)}개 최종 선택")

    # ── 최종 저장 (미래 관찰일 포함) ──────────────────────────────
    print("\n[4] 최종 데이터셋 저장")
    # 선택된 피처를 전체 df(미래 행 포함)에 적용 후 타겟 컬럼과 합쳐 저장
    df_all_feats = df[feat_cols].ffill()[selected]
    df_final = df_all_feats.join(df[[PRIMARY_TARGET]])

    n_labeled = df_final[PRIMARY_TARGET].notna().sum()
    n_future  = df_final[PRIMARY_TARGET].isna().sum()

    df_final.to_csv(FEATURES_PATH)
    print(f"  → 저장: {FEATURES_PATH}")
    print(f"  라벨 확정: {n_labeled}개  |  미래 예측 대상: {n_future}개 (타겟 NaN)")

    print(f"\n  선택된 피처 ({len(selected)}개):")
    for i, f in enumerate(selected, 1):
        print(f"    {i:>2}. {f}")

    print("  Step 4 완료.")


if __name__ == "__main__":
    main()
