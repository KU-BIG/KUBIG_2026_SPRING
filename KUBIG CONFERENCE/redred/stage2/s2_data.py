"""
s2_data.py — Step 2: 피처 데이터 수집
=======================================
관찰일(obs_date) 기준으로 A~E 그룹 피처를 수집한다.
모든 피처는 관찰일 시점에서 이미 알 수 있는 값만 사용 (lookahead 없음).

[A] 기술적 지표       — SK하이닉스 가격·모멘텀·변동성
[B] 시장 센티먼트     — VIX, SOX, NVDA, TSM, ASML, Samsung, S&P500
[C] WSTS 역사 데이터  — Worldwide·AP YoY%, 이동평균, 모멘텀, 사이클 위치
[D] FRED 거시지표     — 금리차, 기준금리, 산업생산, PCE, 소비자심리
[E] 환율·원자재       — USD/KRW, WTI 유가

입력:  outputs/data/quarterly_dates.csv  (s1_dates.py 출력)
출력:  outputs/data/raw_quarterly.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings("ignore")

from config import DATES_PATH, RAW_PATH, WSTS_PATH, FRED_API_KEY, START_YEAR, END_YEAR

TICKERS = {
    "SKH":     "000660.KS",
    "Samsung": "005930.KS",
    "SOX":     "^SOX",
    "NVDA":    "NVDA",
    "TSM":     "TSM",
    "ASML":    "ASML",
    "SPX":     "^GSPC",
    "VIX":     "^VIX",
    "Oil":     "CL=F",
    "USDKRW":  "KRW=X",
}

FRED_SERIES = {
    "T10Y2Y":    "T10Y2Y",
    "FedFunds":  "DFF",
    "IndProd":   "INDPRO",
    "PCE_Core":  "PCEPILFE",
    "ConsSenti": "UMCSENT",
    "T10Y3M":    "T10Y3M",
}


# ──────────────────────────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────────────────────────

def fetch_daily_prices(start: str, end: str) -> dict:
    print("[yfinance] 일봉 수집 중...")
    prices = {}
    for name, ticker in TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                              interval="1d", auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  ✗ {name}: 데이터 없음")
                continue
            close = raw["Close"].squeeze()
            close.index = pd.to_datetime(close.index)
            prices[name] = close
            print(f"  ✓ {name}: {len(close)}일")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return prices


def fetch_fred_monthly(start: str, end: str) -> pd.DataFrame:
    if not FRED_API_KEY:
        print("[FRED] API 키 미설정 — 건너뜀")
        return pd.DataFrame()
    print("[FRED] 월별 거시지표 수집 중...")
    fred = Fred(api_key=FRED_API_KEY)
    dfs = {}
    for name, sid in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s.index = pd.to_datetime(s.index)
            s = s.resample("ME").last()
            dfs[name] = s
            print(f"  ✓ {name} ({sid}): {len(s)}개월")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return pd.DataFrame(dfs) if dfs else pd.DataFrame()


def load_wsts_monthly(start: str) -> pd.DataFrame:
    """WSTS 엑셀에서 직접 파싱 (Stage 1 import 충돌 방지)."""
    df_raw = pd.read_excel(WSTS_PATH, sheet_name="Monthly Data", header=None)

    month_cols    = list(range(1, 13))
    month_names   = ["January","February","March","April","May","June",
                     "July","August","September","October","November","December"]
    regions_order = ["Americas", "Europe", "Japan", "Asia Pacific", "Worldwide"]

    records, current_year = [], None
    for _, row in df_raw.iterrows():
        cell0 = row.iloc[0]
        if isinstance(cell0, (int, float)) and not pd.isna(cell0) and 1980 <= int(cell0) <= 2030:
            current_year = int(cell0)
            continue
        if isinstance(cell0, str) and cell0.strip() in regions_order:
            region = cell0.strip()
            for m_idx, _ in zip(month_cols, month_names):
                val = row.iloc[m_idx]
                if pd.notna(val) and current_year is not None:
                    date = pd.Timestamp(year=current_year, month=m_idx, day=1) + pd.offsets.MonthEnd(0)
                    records.append({"date": date, "region": region, "revenue": float(val)})

    df_long = pd.DataFrame(records)
    df_long["region"] = df_long["region"].str.replace(" ", "_")
    df_wide = df_long.pivot_table(index="date", columns="region", values="revenue")
    df_wide.columns.name = None
    df_wide = df_wide.sort_index()
    df_wide = df_wide[df_wide.index >= pd.Timestamp(start)]
    return df_wide


# ──────────────────────────────────────────────────────────────
# 스냅 헬퍼 (lookahead 없음: date 이전 값만 사용)
# ──────────────────────────────────────────────────────────────

def get_price_on(series: pd.Series, date: pd.Timestamp, window: int = 5) -> float:
    """date 이전 window 영업일 이내 가장 최근 종가."""
    cutoff = date - pd.Timedelta(days=window * 2)
    window_s = series[(series.index >= cutoff) & (series.index <= date)]
    return float(window_s.iloc[-1]) if not window_s.empty else np.nan


def get_last_monthly(series: pd.Series, date: pd.Timestamp) -> float:
    """date 이전 가장 최근 월말 값."""
    window = series[series.index <= date]
    return float(window.iloc[-1]) if not window.empty else np.nan


def compute_return(prices: pd.Series, date: pd.Timestamp, months: int) -> float:
    """date 기준 months개월 전 대비 수익률 (%)."""
    p_now  = get_price_on(prices, date)
    past_d = date - pd.DateOffset(months=months)
    p_past = get_price_on(prices, past_d)
    if np.isnan(p_now) or np.isnan(p_past) or p_past == 0:
        return np.nan
    return (p_now / p_past - 1) * 100


def compute_vol(prices: pd.Series, date: pd.Timestamp, days: int = 60) -> float:
    """연환산 실현 변동성 (%)."""
    start = date - pd.Timedelta(days=days + 15)
    w = prices[(prices.index >= start) & (prices.index <= date)]
    if len(w) < 20:
        return np.nan
    return float(w.pct_change().dropna().std() * np.sqrt(252) * 100)


def compute_rsi(prices: pd.Series, date: pd.Timestamp, period: int = 14) -> float:
    start = date - pd.Timedelta(days=period * 4)
    w = prices[(prices.index >= start) & (prices.index <= date)].dropna()
    if len(w) < period + 1:
        return np.nan
    delta = w.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).iloc[-1]
    return float(rsi)


def compute_ma_ratio(prices: pd.Series, date: pd.Timestamp, days: int) -> float:
    """현재가 / MA(days) - 1, %."""
    start = date - pd.Timedelta(days=days + 30)
    w = prices[(prices.index >= start) & (prices.index <= date)]
    if len(w) < days // 2:
        return np.nan
    p_now = float(w.iloc[-1])
    ma    = float(w.iloc[-days:].mean())
    return (p_now / ma - 1) * 100 if ma > 0 else np.nan


def compute_52w_pct(prices: pd.Series, date: pd.Timestamp) -> float:
    """52주 고저 내 현재가 위치 (0~100%)."""
    start = date - pd.Timedelta(days=365)
    w = prices[(prices.index >= start) & (prices.index <= date)]
    if len(w) < 20:
        return np.nan
    p_now = float(w.iloc[-1])
    hi, lo = w.max(), w.min()
    return (p_now - lo) / (hi - lo + 1e-9) * 100


def cycle_position(series: pd.Series, date: pd.Timestamp, window: int = 24) -> float:
    """window개월 내 백분위 위치 (0~1)."""
    w = series[series.index <= date].iloc[-window:]
    if len(w) < window // 2:
        return np.nan
    v = w.iloc[-1]
    return float((v - w.min()) / (w.max() - w.min() + 1e-9))


# ──────────────────────────────────────────────────────────────
# 관찰일 피처 스냅
# ──────────────────────────────────────────────────────────────

def snap_features(obs_date: pd.Timestamp, prices: dict,
                  fred: pd.DataFrame, wsts: pd.DataFrame) -> dict:
    feat = {}

    # ── A. SK하이닉스 기술적 지표 ──────────────────────────────
    skh = prices.get("SKH")
    if skh is not None:
        p = get_price_on(skh, obs_date)
        feat["SKH_price_obs"]     = p
        feat["SKH_log_price_obs"] = np.log(p) if (p and p > 0) else np.nan
        for m in [1, 3, 6, 12]:
            feat[f"SKH_ret_{m}m"] = compute_return(skh, obs_date, m)
        feat["SKH_vol_60d"]       = compute_vol(skh, obs_date, 60)
        feat["SKH_RSI_14"]        = compute_rsi(skh, obs_date, 14)
        feat["SKH_vs_ma60"]       = compute_ma_ratio(skh, obs_date, 60)
        feat["SKH_vs_ma120"]      = compute_ma_ratio(skh, obs_date, 120)
        feat["SKH_52w_pct"]       = compute_52w_pct(skh, obs_date)

    # ── B. 시장 센티먼트 ───────────────────────────────────────
    vix = prices.get("VIX")
    if vix is not None:
        feat["VIX_level"]  = get_price_on(vix, obs_date)
        feat["VIX_chg_1m"] = compute_return(vix, obs_date, 1)

    for name in ["SOX", "NVDA", "TSM", "ASML", "Samsung", "SPX"]:
        p = prices.get(name)
        if p is not None:
            for m in [1, 3, 6]:
                feat[f"{name}_ret_{m}m"] = compute_return(p, obs_date, m)

    sox_3m = feat.get("SOX_ret_3m")
    spx_3m = feat.get("SPX_ret_3m")
    if sox_3m is not None and spx_3m is not None:
        feat["SOX_vs_SPX_3m"] = sox_3m - spx_3m

    # ── C. WSTS 실제 역사 데이터 ───────────────────────────────
    if not wsts.empty and "Worldwide" in wsts.columns:
        ww     = wsts["Worldwide"]
        ww_yoy = ww.pct_change(12) * 100

        feat["WSTS_WW_YoY"]          = get_last_monthly(ww_yoy, obs_date)
        feat["WSTS_WW_YoY_ma3"]      = get_last_monthly(ww_yoy.rolling(3).mean(), obs_date)
        feat["WSTS_WW_YoY_ma6"]      = get_last_monthly(ww_yoy.rolling(6).mean(), obs_date)
        feat["WSTS_WW_YoY_mom_3_12"] = get_last_monthly(
            ww_yoy.diff(3) - ww_yoy.diff(12), obs_date
        )
        feat["WSTS_WW_cycle_pos"]    = cycle_position(ww_yoy, obs_date, 24)

        if "Asia_Pacific" in wsts.columns:
            ap_yoy = wsts["Asia_Pacific"].pct_change(12) * 100
            feat["WSTS_AP_YoY"]   = get_last_monthly(ap_yoy, obs_date)
            feat["WSTS_AP_YoY_ma3"] = get_last_monthly(ap_yoy.rolling(3).mean(), obs_date)

    # ── D. FRED 거시지표 ───────────────────────────────────────
    if not fred.empty:
        for col in ["T10Y2Y", "FedFunds", "T10Y3M"]:
            if col in fred.columns:
                feat[col]           = get_last_monthly(fred[col], obs_date)
                feat[f"{col}_chg3"] = get_last_monthly(fred[col].diff(3), obs_date)

        for col in ["IndProd", "PCE_Core"]:
            if col in fred.columns:
                yoy = fred[col].pct_change(12) * 100
                feat[f"{col}_YoY"] = get_last_monthly(yoy, obs_date)

        if "ConsSenti" in fred.columns:
            feat["ConsSenti"]      = get_last_monthly(fred["ConsSenti"], obs_date)
            feat["ConsSenti_chg3"] = get_last_monthly(fred["ConsSenti"].diff(3), obs_date)

        # 금리 역전 여부
        if "T10Y3M" in fred.columns:
            t3m = fred["T10Y3M"]
            val = get_last_monthly(t3m, obs_date)
            feat["T10Y3M_inverted"] = int(val < 0) if not np.isnan(val) else np.nan

    # ── E. 환율·원자재 ─────────────────────────────────────────
    for name in ["Oil", "USDKRW"]:
        p = prices.get(name)
        if p is not None:
            feat[f"{name}_ret_3m"] = compute_return(p, obs_date, 3)
            feat[f"{name}_ret_6m"] = compute_return(p, obs_date, 6)

    return feat


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Step 2  피처 데이터 수집")
    print("=" * 64)

    df_dates  = pd.read_csv(DATES_PATH, parse_dates=["obs_date", "earnings_date"])
    obs_dates = sorted(df_dates["obs_date"].dropna().unique())
    print(f"\n  관찰일 {len(obs_dates)}개 처리")

    start_str = f"{START_YEAR - 2}-01-01"
    end_str   = f"{END_YEAR + 1}-01-01"

    print()
    prices = fetch_daily_prices(start_str, end_str)
    print()
    fred   = fetch_fred_monthly(start_str, end_str)
    print()
    wsts   = load_wsts_monthly(start_str)
    print(f"[WSTS] 로드: {len(wsts)}개월")

    print("\n[관찰일별 피처 스냅 중...]")
    rows = []
    for obs_d in obs_dates:
        feat = snap_features(pd.Timestamp(obs_d), prices, fred, wsts)
        feat["obs_date"] = obs_d
        rows.append(feat)

    df_raw = pd.DataFrame(rows).set_index("obs_date")
    df_raw.index = pd.to_datetime(df_raw.index)

    print(f"\n  피처 수집 완료: {len(df_raw)}개 관찰 × {len(df_raw.columns)}개 피처")
    print(f"  NaN 비율: {df_raw.isna().mean().mean():.1%}")

    df_raw.to_csv(RAW_PATH)
    print(f"  → 저장: {RAW_PATH}")
    print("  Step 2 완료.")


if __name__ == "__main__":
    main()
