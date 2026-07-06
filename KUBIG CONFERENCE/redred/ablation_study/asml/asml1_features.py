"""
asml1_features.py — Step 1: ASML 전용 피처 매트릭스 빌드

피처 구성:
  macro(6):      FedFunds_lag6, FRED_T10Y2Y_lag6, FRED_ConsSenti_YoY_lag6,
                 FedFunds_diff12, FRED_T10Y2Y_chg3, Worldwide_YoY_ma12
  SOX(3):        Ret_SOX_lag6, Ret_SOX_ma6, Ret_SOX_vol6
  FX EUR(2):     usdeur_lag6, usdeur_chg6_lag6 (yfinance "EURUSD=X" → 역수)
  PCE PC(1):     pce_computers_yoy_lag6 (수집 실패 시 스킵)
  semicap(1):    semicap_yoy_lag6 (FRED CAPG3344S, 수집 실패 시 스킵)
  semicapu(1):   semicapu_lag6 (FRED CAPUTLG3344S, 수준값 lag6, 수집 실패 시 스킵)
  supply(1):     wsts_pred_t6
  ASML self(3):  ASML_return_lag6, ASML_return_lag12, ASML_vol_lag6
  target(1):     ASML_fwd6 (피처 사용 금지)

leakage 방지:
  - MACRO/SOX: features_dataset.csv에서 lag 이미 적용됨
  - FX EUR: EURUSD=X 역수 후 shift(6) / pct_change(6).shift(6)
  - PCE/semicap: pct_change(12) 후 shift(6)
  - semicapu: 수준값 shift(6) (가동률은 이미 퍼센트 단위)
  - ASML self: shift(6), shift(12), rolling(6).std().shift(6)
  - ASML_fwd6: shift(-6) — 타겟 전용, 피처 절대 사용 금지
"""

import sys
import os

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asml_config import (
    FEATURES_PATH, OOS_PRED_PATH, ASML_FEATURES_PATH,
    MACRO_COLS, SOX_COLS, FRED_API_KEY,
    TARGET_COL, SUPPLY_COL,
)


def load_base_macro_sox() -> pd.DataFrame:
    df_feat = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    missing = [c for c in MACRO_COLS + SOX_COLS if c not in df_feat.columns]
    if missing:
        raise ValueError(f"features_dataset.csv에 컬럼 없음: {missing}")
    df_base = df_feat[MACRO_COLS + SOX_COLS].copy()
    print(f"[macro+SOX] {df_base.shape}, "
          f"{df_base.index[0].strftime('%Y-%m')} ~ {df_base.index[-1].strftime('%Y-%m')}")
    return df_base


def load_supply() -> pd.DataFrame:
    df = pd.read_parquet(OOS_PRED_PATH)
    print(f"[supply] wsts_pred_t6: {df.shape}, "
          f"{df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
    return df


def load_fx_eur() -> pd.DataFrame:
    # EURUSD=X: EUR/USD → 역수 USD/EUR 로 변환
    raw = yf.download("EURUSD=X", start="1993-01-01", auto_adjust=True, progress=False)["Close"]
    eurusd_m = raw.resample("ME").last()
    if isinstance(eurusd_m, pd.DataFrame):
        eurusd_m = eurusd_m.squeeze()
    eurusd_m = eurusd_m.dropna()

    usdeur_m = 1 / eurusd_m  # USD/EUR (역수): 값 클수록 USD 강세

    df_fx = pd.DataFrame({
        "usdeur_lag6":      usdeur_m.shift(6),                      # 수준값 lag6: leakage 없음
        "usdeur_chg6_lag6": usdeur_m.pct_change(6).shift(6) * 100,  # 6개월 변화율 lag6: leakage 없음
    }, index=usdeur_m.index)

    print(f"[FX EUR] usdeur: {df_fx.dropna().shape[0]}행 (lag 적용 후)")
    return df_fx


def load_pce_computers() -> pd.Series | None:
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        pce_raw = fred.get_series("DNDGRG3M086SBEA", observation_start="1993-01-01")
        pce_m   = pce_raw.resample("ME").last()
        pce_yoy = pce_m.pct_change(12) * 100   # YoY%: leakage 없음
        pce_lag6 = pce_yoy.shift(6)             # lag6: leakage 없음
        pce_lag6.name = "pce_computers_yoy_lag6"
        print(f"[PCE PC] pce_computers_yoy_lag6 수집 완료 ({pce_lag6.dropna().shape[0]}행)")
        return pce_lag6
    except Exception as e:
        print(f"[경고] PCE computers(DNDGRG3M086SBEA) 수집 실패: {e}")
        print("  → pce_computers_yoy_lag6 스킵, 나머지 피처로 진행")
        return None


def load_semicap() -> pd.Series | None:
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        raw  = fred.get_series("CAPG3344S", observation_start="1993-01-01")
        m    = raw.resample("ME").last()
        yoy  = m.pct_change(12) * 100   # YoY%: leakage 없음
        lag6 = yoy.shift(6)             # lag6: leakage 없음
        lag6.name = "semicap_yoy_lag6"
        print(f"[semicap] semicap_yoy_lag6 수집 완료 ({lag6.dropna().shape[0]}행)")
        return lag6
    except Exception as e:
        print(f"[경고] semicap(CAPG3344S) 수집 실패: {e}")
        print("  → semicap_yoy_lag6 스킵, 나머지 피처로 진행")
        return None


def load_semicapu() -> pd.Series | None:
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        raw  = fred.get_series("CAPUTLG3344S", observation_start="1993-01-01")
        m    = raw.resample("ME").last()
        # 가동률은 퍼센트 수준값 그대로 lag6 (YoY 변환 없음)
        lag6 = m.shift(6)               # 수준값 lag6: leakage 없음
        lag6.name = "semicapu_lag6"
        print(f"[semicapu] semicapu_lag6 수집 완료 ({lag6.dropna().shape[0]}행)")
        return lag6
    except Exception as e:
        print(f"[경고] semicapu(CAPUTLG3344S) 수집 실패: {e}")
        print("  → semicapu_lag6 스킵, 나머지 피처로 진행")
        return None


def load_asml_price() -> pd.DataFrame:
    raw = yf.download("ASML", start="1993-01-01", auto_adjust=True, progress=False)["Close"]
    price_m = raw.resample("ME").last()
    if isinstance(price_m, pd.DataFrame):
        price_m = price_m.squeeze()
    price_m = price_m.dropna()

    monthly_ret = price_m.pct_change() * 100

    # 타겟: (price[t+6] - price[t]) / price[t] × 100 (미래값 → 피처 사용 금지)
    ASML_fwd6 = price_m.pct_change(6).shift(-6) * 100

    df_asml = pd.DataFrame({
        TARGET_COL:           ASML_fwd6,
        "ASML_return_lag6":   monthly_ret.shift(6),                   # lag6: leakage 없음
        "ASML_return_lag12":  monthly_ret.shift(12),                  # lag12: leakage 없음
        "ASML_vol_lag6":      monthly_ret.rolling(6).std().shift(6),  # lag6: leakage 없음
    }, index=price_m.index)

    print(f"[ASML] {df_asml.shape}, "
          f"{df_asml.index[0].strftime('%Y-%m')} ~ {df_asml.index[-1].strftime('%Y-%m')}")
    return df_asml


def main() -> None:
    print("=" * 55)
    print("  asml1_features.py — ASML 피처 매트릭스 빌드")
    print("=" * 55)

    df_base      = load_base_macro_sox()
    df_supply    = load_supply()
    df_fx        = load_fx_eur()
    pce_lag6     = load_pce_computers()
    semi_lag6    = load_semicap()
    semicapu_lag6 = load_semicapu()
    df_asml      = load_asml_price()

    # 통합 join
    df = df_base.join(df_supply, how="inner")
    df = df.join(df_fx, how="left")                                  # FX: 항상 존재
    if pce_lag6      is not None: df = df.join(pce_lag6,      how="left")
    if semi_lag6     is not None: df = df.join(semi_lag6,     how="left")
    if semicapu_lag6 is not None: df = df.join(semicapu_lag6, how="left")
    df = df.join(df_asml, how="inner").dropna()

    print(f"\n[통합 결과] {df.shape}")
    print(f"  날짜 범위: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
    print(f"  컬럼 목록: {list(df.columns)}")
    print(f"  NaN 수: {df.isna().sum().sum()}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    print(f"\n  피처 {len(feature_cols)}개: {feature_cols}")
    print(f"  타겟: {TARGET_COL}")

    df.to_parquet(ASML_FEATURES_PATH)
    print(f"\n저장: {ASML_FEATURES_PATH}")
    print("완료.")


if __name__ == "__main__":
    main()
