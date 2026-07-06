"""
sk1_price.py — Step 1: SK하이닉스 월별 주가 수집
=================================================
yfinance로 000660.KS (SK하이닉스)의 월말 종가를 수집하고
타겟 1종 + SK 자체 피처 3종 + 환율 피처 2종을 생성한다.

타겟 (미래값 — 피처로 절대 사용 금지):
  hynix_fwd6 = (price[t+6] - price[t]) / price[t] × 100  (6개월 forward return)

SK 자체 피처 (모두 t 이전 정보, leakage 없음):
  hynix_return_lag6  ← 6개월 전 월별 수익률 (모멘텀)
  hynix_return_lag12 ← 12개월 전 월별 수익률 (계절성)
  hynix_vol_lag6     ← 6개월 rolling 변동성(std), lag6 적용

환율 피처 (USD/KRW, leakage 없음 — lag6, lag12 적용):
  usdkrw_lag6      ← 6개월 전 환율 수준 (원화 강약)
  usdkrw_chg6_lag6 ← (usdkrw[t-6] - usdkrw[t-12]) / usdkrw[t-12] × 100

출력:  skhynix/outputs/data/hynix_price.parquet
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

import pandas as pd
import yfinance as yf

from sk_config import PRICE_PATH


def main():
    print("=" * 64)
    print("  Step 1  SK하이닉스 월별 주가 수집")
    print("=" * 64)

    print("\n[1] yfinance 다운로드 (000660.KS, 1996-01~)")
    raw = yf.download(
        "000660.KS",
        start="1996-01-01",
        auto_adjust=True,
        progress=False,
    )["Close"]
    hynix = raw.squeeze().rename("hynix_price")

    print(f"  일별 hynix: {hynix.dropna().shape[0]}행  "
          f"{hynix.dropna().index[0].strftime('%Y-%m')} ~ {hynix.dropna().index[-1].strftime('%Y-%m')}")

    print("\n[2] 월말 종가 집계")
    hynix_m = hynix.resample("ME").last()
    df = hynix_m.to_frame().dropna()
    print(f"  월별: {df.shape[0]}행  "
          f"{df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")

    print("\n[3] 타겟 계산 (leakage 주의: 미래값이므로 피처로 사용 금지)")
    # 6개월 수익률을 shift(-6)으로 forward 계산
    df["hynix_fwd6"] = df["hynix_price"].pct_change(periods=6).shift(-6) * 100

    print("\n[4] SK 자체 피처 생성 (모두 t 이전 값 — leakage 없음)")
    monthly_ret = df["hynix_price"].pct_change() * 100           # 월별 수익률
    vol6        = monthly_ret.rolling(6).std()                    # 6개월 rolling std
    df["hynix_return_lag6"]  = monthly_ret.shift(6)              # 6개월 전 수익률
    df["hynix_return_lag12"] = monthly_ret.shift(12)             # 12개월 전 수익률
    df["hynix_vol_lag6"]     = vol6.shift(6)                     # 6개월 변동성의 6개월 전값

    print("\n[4-2] USD/KRW 환율 피처 생성 (leakage 없음 — lag6, lag12 적용)")
    try:
        usdkrw_raw = yf.download("KRW=X", start="1996-01-01", auto_adjust=True, progress=False)["Close"]
        usdkrw_m   = usdkrw_raw.resample("ME").last()
        # usdkrw_lag6: t 시점 피처 = t-6 시점 환율 (leakage 없음)
        df["usdkrw_lag6"]      = usdkrw_m.shift(6).reindex(df.index)
        # usdkrw_chg6_lag6: (usdkrw[t-6] - usdkrw[t-12]) / usdkrw[t-12] × 100 (leakage 없음)
        df["usdkrw_chg6_lag6"] = (
            (usdkrw_m.shift(6) - usdkrw_m.shift(12)) / usdkrw_m.shift(12) * 100
        ).reindex(df.index)
        new_nan = df[["usdkrw_lag6", "usdkrw_chg6_lag6"]].isnull().sum()
        print(f"  신규 컬럼 NaN 현황: {dict(new_nan)}")
    except Exception as e:
        print(f"  [경고] KRW=X 수집 실패 ({e}) → usdkrw 피처 스킵")

    print("\n[5] NaN 제거 (마지막 6개월 타겟 NaN 포함)")
    before = len(df)
    df = df.dropna()
    print(f"  {before}행 → {len(df)}행 ({before - len(df)}행 제거)")

    print("\n[6] 저장")
    df.to_parquet(PRICE_PATH)
    print(f"  → 저장: {PRICE_PATH}")
    print(f"  Shape:  {df.shape}")
    print(f"  날짜:   {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
    print(f"  컬럼:   {list(df.columns)}")
    nan_count = df.isnull().sum()
    if nan_count.any():
        print(f"  NaN 현황:\n{nan_count[nan_count > 0]}")
    else:
        print("  NaN: 없음")
    print("  Step 1 완료.")


if __name__ == "__main__":
    main()
