"""
sk2_features.py — Step 2: 월별 피처 매트릭스 빌드
===================================================
네 소스를 날짜 기준으로 inner join해 최종 피처 데이터셋을 생성한다.

입력:
  model/outputs/data/features_dataset.csv     — 월별 거시경제 피처 (Stage 1)
  skhynix/outputs/data/wsts_oos_preds.parquet — 월별 wsts_pred_t6 (Step 0)
  skhynix/outputs/data/hynix_price.parquet    — 타겟 + SK/환율 피처 (Step 1)
  pykrx API                                   — 외국인 순매수 (월별)

leakage 체크:
  - wsts_pred_t6: expanding walk-forward OOS 예측값. leakage 없음.
  - 거시경제 피처: features_dataset에서 _lag6/_lag12 등 이미 적용됨. leakage 없음.
  - SK 자체 피처: hynix_return_lag6, hynix_return_lag12, hynix_vol_lag6. leakage 없음.
  - usdkrw_lag6, usdkrw_chg6_lag6: sk1에서 lag6·lag12 적용됨. leakage 없음.
  - foreign_net_lag6, foreign_net_chg6_lag6: 월별 합계 후 lag6 적용. leakage 없음.
  - hynix_fwd6: 미래값. 피처 컬럼으로 절대 사용 금지.

출력:  skhynix/outputs/data/stage2_features.parquet
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

import pandas as pd

from sk_config import (
    FEATURES_PATH, OOS_PRED_PATH, PRICE_PATH, STAGE2_PATH,
    MACRO_FEATURE_CANDIDATES,
)


def main():
    print("=" * 64)
    print("  Step 2  월별 피처 매트릭스 빌드")
    print("=" * 64)

    print("\n[1] Stage 1 거시경제 피처 로드 + 후보 컬럼 선택")
    macro_df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    found, skipped = [], []
    for col in MACRO_FEATURE_CANDIDATES:
        if col in macro_df.columns:
            found.append(col)
        else:
            skipped.append(col)
    if skipped:
        print(f"  [경고] 미존재 피처 스킵: {skipped}")
    print(f"  사용 피처: {found}")
    macro_sub = macro_df[found].copy()

    print("\n[2] OOS 예측값 로드")
    oos = pd.read_parquet(OOS_PRED_PATH)
    print(f"  Shape: {oos.shape}  날짜: {oos.index[0].strftime('%Y-%m')} ~ {oos.index[-1].strftime('%Y-%m')}")

    print("\n[3] 주가/타겟 데이터 로드")
    price = pd.read_parquet(PRICE_PATH)
    # 피처로 사용할 컬럼만 분리 (타겟은 join 후 별도 관리)
    sk_feature_cols = ["hynix_return_lag6", "hynix_return_lag12", "hynix_vol_lag6"]
    # 환율 피처: sk1에서 lag 적용 완료 (leakage 없음) — 존재할 때만 포함
    usdkrw_cols = [c for c in ["usdkrw_lag6", "usdkrw_chg6_lag6"] if c in price.columns]
    if usdkrw_cols:
        sk_feature_cols += usdkrw_cols
        print(f"  환율 피처 포함: {usdkrw_cols}")
    else:
        print("  [경고] 환율 피처 미발견 → 스킵")
    target_cols = ["hynix_fwd6"]
    print(f"  Shape: {price.shape}  날짜: {price.index[0].strftime('%Y-%m')} ~ {price.index[-1].strftime('%Y-%m')}")

    print("\n[4] 날짜 기준 Inner Join")
    df = macro_sub.join(oos, how="inner")
    df = df.join(price[sk_feature_cols + target_cols], how="inner")
    print(f"  교집합: {df.shape[0]}행  날짜: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")

    print("\n[4-2] 외국인 순매수 피처 생성 (pykrx, leakage 없음 — lag6, lag12 적용)")
    try:
        try:
            from pykrx import stock as pykrx_stock
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "pykrx", "-q"], check=True)
            from pykrx import stock as pykrx_stock

        print("  pykrx 데이터 수집 중 (약 10-30초 소요)...")
        start_str = "20010101"
        end_str   = df.index[-1].strftime("%Y%m%d")
        df_inv = pykrx_stock.get_market_trading_value_by_date(start_str, end_str, "000660")

        # 외국인 순매수 컬럼 탐색
        foreign_col = None
        for candidate in ["외국인합계", "외국인", "외국인_순매수"]:
            if candidate in df_inv.columns:
                foreign_col = candidate
                break
        if foreign_col is None:
            raise ValueError(f"외국인 컬럼 미발견. 컬럼: {list(df_inv.columns)}")

        # 월별 합계 (억원)
        foreign_monthly = (df_inv[foreign_col].resample("ME").sum() / 1e8).rename("foreign_net_raw")

        # lag 피처 생성 (leakage 없음: t 시점 피처 = t-6 이전 값 사용)
        foreign_features = pd.DataFrame({
            "foreign_net_lag6":      foreign_monthly.shift(6),           # t-6 시점 순매수
            "foreign_net_chg6_lag6": foreign_monthly.shift(6) - foreign_monthly.shift(12),  # 모멘텀
        })
        df = df.join(foreign_features, how="left")
        new_nan = df[["foreign_net_lag6", "foreign_net_chg6_lag6"]].isnull().sum()
        print(f"  외국인 순매수 피처 추가 완료. NaN: {dict(new_nan)}")
    except Exception as e:
        print(f"  [경고] pykrx 수집 실패 ({e}) → foreign_net 피처 2종 스킵")

    print("\n[5] NaN 제거")
    before = len(df)
    df = df.dropna()
    print(f"  {before}행 → {len(df)}행 ({before - len(df)}행 제거)")

    print("\n[6] 저장")
    df.to_parquet(STAGE2_PATH)
    print(f"  → 저장: {STAGE2_PATH}")
    print(f"  Shape:  {df.shape}")
    print(f"  날짜:   {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
    print(f"\n  컬럼 목록:")
    feature_cols = [c for c in df.columns if c not in target_cols]
    print(f"  [피처 {len(feature_cols)}개] {feature_cols}")
    print(f"  [타겟 {len(target_cols)}개] {target_cols}")
    nan_count = df.isnull().sum()
    if nan_count.any():
        print(f"\n  NaN 현황:\n{nan_count[nan_count > 0]}")
    else:
        print("\n  NaN: 없음")
    print("  Step 2 완료.")


if __name__ == "__main__":
    main()
