"""
s1_dates.py — Step 1: 분기 날짜 생성 + 타겟 수익률 산출
=========================================================
SK하이닉스 분기 실적발표일(Target)과 관찰일(Observation)을 생성하고
두 날짜의 종가로 6개월 종가 수익률을 계산한다.

날짜 규칙:
  Target 날짜   : 매년 1·4·7·10월 넷째 주 목요일 (분기 실적발표일)
  Observation 날짜: Target 날짜 정확히 6개월 전 같은 요일
    → 1월 → 전년 7월 / 4월 → 전년 10월 / 7월 → 1월 / 10월 → 4월

타겟:
  TARGET_SKH_6M_RET = (P_earnings / P_obs - 1) × 100  (%)

입력:  없음 (yfinance 수집)
출력:  outputs/data/quarterly_dates.csv
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from config import DATES_PATH, START_YEAR, END_YEAR, PRIMARY_TARGET

EARNINGS_MONTHS = [1, 4, 7, 10]
SKH_TICKER      = "000660.KS"


def get_4th_thursday(year: int, month: int):
    """해당 연월의 넷째 주 목요일 반환."""
    d = datetime(year, month, 1)
    days_until_thu = (3 - d.weekday()) % 7   # 3 = 목요일
    first_thu = d + timedelta(days=days_until_thu)
    return (first_thu + timedelta(weeks=3)).date()


def build_date_pairs(start_year: int, end_year: int) -> pd.DataFrame:
    """(obs_date, earnings_date) 쌍 전체 생성."""
    rows = []
    for year in range(start_year, end_year + 1):
        for em in EARNINGS_MONTHS:
            obs_month = em - 6
            obs_year  = year
            if obs_month <= 0:
                obs_month += 12
                obs_year  -= 1
            rows.append({
                "earnings_date": get_4th_thursday(year, em),
                "obs_date":      get_4th_thursday(obs_year, obs_month),
            })
    df = pd.DataFrame(rows).sort_values("obs_date").reset_index(drop=True)
    return df


def get_price_near(prices: pd.Series, target_date, window: int = 5) -> float:
    """target_date 전후 window 영업일 이내 가장 가까운 종가 반환."""
    target_ts = pd.Timestamp(target_date)
    candidates = []
    idx = prices.index.searchsorted(target_ts)
    for offset in range(-window, window + 1):
        pos = idx + offset
        if 0 <= pos < len(prices):
            gap  = abs((prices.index[pos] - target_ts).days)
            candidates.append((gap, float(prices.iloc[pos])))
    return min(candidates, key=lambda x: x[0])[1] if candidates else np.nan


def main():
    print("=" * 64)
    print("  Step 1  분기 날짜 생성 + 타겟 수익률 산출")
    print("=" * 64)

    print("\n[1] 날짜 쌍 생성")
    df = build_date_pairs(START_YEAR, END_YEAR)
    print(f"  총 {len(df)}개 관찰 포인트  "
          f"({df['obs_date'].min()} ~ {df['earnings_date'].max()})")

    print("\n[2] SK하이닉스 일봉 수집 (000660.KS)")
    raw = yf.download(
        SKH_TICKER,
        start=f"{START_YEAR - 1}-01-01",
        end=f"{END_YEAR + 1}-01-01",
        interval="1d", auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"SK하이닉스({SKH_TICKER}) 데이터 수집 실패")

    close = raw["Close"].squeeze()
    close.index = pd.to_datetime(close.index)
    print(f"  일봉 수집: {len(close)}일  "
          f"({close.index.min().date()} ~ {close.index.max().date()})")

    print("\n[3] 종가 스냅 + 수익률 계산")
    prices_obs, prices_earnings = [], []
    for _, row in df.iterrows():
        prices_obs.append(get_price_near(close, row["obs_date"]))
        prices_earnings.append(get_price_near(close, row["earnings_date"]))

    df["price_obs"]      = prices_obs
    df["price_earnings"] = prices_earnings
    df[PRIMARY_TARGET]   = (df["price_earnings"] / df["price_obs"] - 1) * 100

    # 미래 날짜(earnings_date가 오늘 이후) → 타겟 NaN
    today = pd.Timestamp.today().date()
    mask  = df["earnings_date"] > today
    df.loc[mask, [PRIMARY_TARGET, "price_earnings"]] = np.nan

    df_valid = df.dropna(subset=["price_obs"])
    df_labeled = df_valid.dropna(subset=[PRIMARY_TARGET])

    bull = (df_labeled[PRIMARY_TARGET] > 0).sum()
    bear = (df_labeled[PRIMARY_TARGET] <= 0).sum()

    print(f"  유효 관찰: {len(df_valid)}개  |  라벨 확정: {len(df_labeled)}개")
    print(f"  Bull(상승): {bull}  /  Bear(하락): {bear}")
    print(f"  수익률 범위: {df_labeled[PRIMARY_TARGET].min():.1f}% ~ "
          f"{df_labeled[PRIMARY_TARGET].max():.1f}%")
    print(f"  수익률 평균: {df_labeled[PRIMARY_TARGET].mean():.1f}%  "
          f"표준편차: {df_labeled[PRIMARY_TARGET].std():.1f}%")

    df_valid.to_csv(DATES_PATH, index=False)
    print(f"\n  → 저장: {DATES_PATH}")
    print("  Step 1 완료.")


if __name__ == "__main__":
    main()
