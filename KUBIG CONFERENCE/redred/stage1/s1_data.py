"""
s1_data.py — Step 1: 데이터 수집 + 피처 엔지니어링
=====================================================
WSTS 엑셀, FRED API, yfinance에서 데이터를 수집하고
6개월 선행 타겟 및 이동평균/변동성/Lag 피처를 생성한다.

입력:  wsts_historical.xlsx (프로젝트 루트)
출력:  outputs/data/merged_dataset.csv
       outputs/data/features_dataset.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings("ignore")

from config import (
    WSTS_PATH, MERGED_PATH, FEATURES_PATH,
    DATA_DIR, FIG_DIR,
    FRED_API_KEY, START_DATE, END_DATE, PRIMARY_TARGET,
)

# ── 피처 엔지니어링 파라미터 ──────────────────────────────────
TARGET_HORIZON  = 6
TARGETS         = ["Worldwide", "Asia_Pacific"]
MA_WINDOWS      = [3, 6, 12]
VOL_WINDOWS     = [3, 6]
LAG_MONTHS      = [6, 12]
EQUITY_COLS     = ["Ret_SOX", "Ret_NVDA", "Ret_TSM",
                   "Ret_ASML", "Ret_Samsung", "Ret_SKHynix"]
FRED_LEVEL_COLS = ["FRED_SemiProd", "FRED_ISM_Mfg", "FRED_T10Y2Y",
                   "FRED_IndProd", "FRED_PCE_Core",
                   "FRED_MfgEmp", "FRED_ConsSenti", "FRED_NewOrder"]


# ──────────────────────────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────────────────────────

def load_wsts() -> pd.DataFrame:
    print("[WSTS] 엑셀 파일 파싱 중...")
    df_raw = pd.read_excel(WSTS_PATH, sheet_name="Monthly Data", header=None)

    month_cols   = list(range(1, 13))
    month_names  = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"]
    regions_order = ["Americas", "Europe", "Japan", "Asia Pacific", "Worldwide"]

    records      = []
    current_year = None

    for _, row in df_raw.iterrows():
        cell0 = row.iloc[0]
        if isinstance(cell0, (int, float)) and not pd.isna(cell0) and 1980 <= int(cell0) <= 2030:
            current_year = int(cell0)
            continue
        if isinstance(cell0, str) and cell0.strip() in regions_order:
            region = cell0.strip()
            for m_idx, m_name in zip(month_cols, month_names):
                val = row.iloc[m_idx]
                if pd.notna(val) and current_year is not None:
                    date = pd.Timestamp(year=current_year, month=m_idx, day=1) + pd.offsets.MonthEnd(0)
                    records.append({"date": date, "region": region,
                                    "revenue_1000usd": float(val)})

    df_long = pd.DataFrame(records)
    if df_long.empty:
        raise ValueError("WSTS 파싱 실패 — 엑셀 구조 확인 필요")

    df_long["region"] = df_long["region"].str.replace(" ", "_")
    df_wide = df_long.pivot_table(index="date", columns="region", values="revenue_1000usd")
    df_wide.columns.name = None
    df_wide = df_wide.sort_index()
    print(f"  완료: {df_wide.shape[0]}개월 × {df_wide.shape[1]}개 지역  "
          f"({df_wide.index.min().date()} ~ {df_wide.index.max().date()})")
    return df_wide


def fetch_fred() -> pd.DataFrame:
    if not FRED_API_KEY:
        print("[FRED] API 키 미설정 — 건너뜀")
        return pd.DataFrame()
    print("[FRED] 거시경제 지표 수집 중...")
    fred = Fred(api_key=FRED_API_KEY)
    series_map = {
        "FRED_SemiProd":  "IPG3344S",
        "FRED_ISM_Mfg":   "NAPM",
        "FRED_T10Y2Y":    "T10Y2Y",
        "FRED_IndProd":   "INDPRO",
        "FRED_PCE_Core":  "PCEPILFE",
        "FRED_MfgEmp":    "MANEMP",
        "FRED_ConsSenti": "UMCSENT",
        "FRED_NewOrder":  "NEWORDER",
        "FRED_T10Y3M":   "T10Y3M",    # 10Y-3M 금리차 (역전 = 경기침체 선행)
        "FRED_InvSales": "ISRATIO",   # 재고/매출 비율 (재고 축적 = 다운사이클 선행)
        "FRED_FedFunds": "DFF",       # 연방기금금리 (통화긴축 사이클)
    }
    dfs = {}
    for col, sid in series_map.items():
        try:
            s = fred.get_series(sid, observation_start=START_DATE, observation_end=END_DATE)
            s.index = pd.to_datetime(s.index)
            s = s.resample("ME").last()
            dfs[col] = s
            print(f"  ✓ {col} ({sid}): {len(s)}개")
        except Exception as e:
            print(f"  ✗ {col}: {e}")
    return pd.DataFrame(dfs) if dfs else pd.DataFrame()


def fetch_equity() -> pd.DataFrame:
    print("[yfinance] 주가 데이터 수집 중...")
    tickers = {"SOX": "^SOX", "NVDA": "NVDA", "TSM": "TSM",
               "ASML": "ASML", "Samsung": "005930.KS", "SKHynix": "000660.KS"}
    price_dfs = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                              interval="1mo", auto_adjust=True, progress=False)
            if raw.empty:
                continue
            close = raw["Close"].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            close.index = close.index.to_period("M").to_timestamp("M")
            price_dfs[f"Price_{name}"] = close
            ret = close.pct_change() * 100
            ret.name = f"Ret_{name}"
            price_dfs[f"Ret_{name}"] = ret
            print(f"  ✓ {name}: {len(close)}개월")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return pd.DataFrame(price_dfs) if price_dfs else pd.DataFrame()


def merge_all(df_wsts, df_fred, df_eq) -> pd.DataFrame:
    def to_me(df):
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
        return df
    df_wsts, df_fred, df_eq = to_me(df_wsts), to_me(df_fred), to_me(df_eq)
    merged = df_wsts.copy()
    for ext in [df_fred, df_eq]:
        if not ext.empty:
            overlap = set(merged.columns) & set(ext.columns)
            merged = merged.join(ext.drop(columns=list(overlap)), how="left")
    merged = merged[merged.index >= pd.Timestamp(START_DATE)].sort_index()
    print(f"[병합] 완료: {merged.shape[0]}개월 × {merged.shape[1]}개 피처")
    return merged


# ──────────────────────────────────────────────────────────────
# 피처 엔지니어링 헬퍼
# ──────────────────────────────────────────────────────────────

def yoy_pct(s):  return s.pct_change(periods=12) * 100
def mom_pct(s):  return s.pct_change() * 100


def add_lags(df, col, lags):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_ma(df, col, windows):
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(w, min_periods=w).mean()
    return df


def add_vol(df, col, windows):
    for w in windows:
        df[f"{col}_vol{w}"] = df[col].rolling(w, min_periods=w).std()
    return df


def add_momentum(df, col):
    df[f"{col}_momentum_3_12"] = df[col].diff(3) - df[col].diff(12)
    return df


def add_accel(df, col):
    df[f"{col}_accel"] = df[col].diff(1)
    return df


def add_cycle_pct(df, col, window=24):
    def prank(x):
        if len(x) < 3: return np.nan
        return (x[-1] - x.min()) / (x.max() - x.min() + 1e-9)
    df[f"{col}_vs_ma24"] = (
        df[col].rolling(window, min_periods=window // 2).apply(prank, raw=True)
    )
    return df


# ──────────────────────────────────────────────────────────────
# 피처 엔지니어링 메인
# ──────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    print("[피처 엔지니어링] 시작...")

    # A. WSTS 지역별 YoY%
    region_cols = [c for c in ["Americas", "Europe", "Japan", "Asia_Pacific", "Worldwide"]
                   if c in df.columns]
    for col in region_cols:
        yc = f"{col}_YoY"
        feat[yc] = yoy_pct(df[col])
        feat = add_lags(feat, yc, LAG_MONTHS)
        feat = add_ma(feat, yc, MA_WINDOWS)
        feat = add_vol(feat, yc, VOL_WINDOWS)
        feat = add_momentum(feat, yc)
        feat = add_accel(feat, yc)
        feat = add_cycle_pct(feat, yc, 24)
    print(f"  A. WSTS YoY 피처: {len(feat.columns)}개")

    # B. 주가 수익률
    eq_exist = [c for c in EQUITY_COLS if c in df.columns]
    for col in eq_exist:
        feat[col] = df[col]
        feat = add_lags(feat, col, LAG_MONTHS)
        feat = add_ma(feat, col, [3, 6])
        feat = add_vol(feat, col, [3, 6])
    if eq_exist:
        feat["Eq_AvgRet"] = feat[[c for c in eq_exist if c in feat.columns]].mean(axis=1)
        feat = add_lags(feat, "Eq_AvgRet", LAG_MONTHS)
    print(f"  B. 주가 수익률 피처 추가 ({len(eq_exist)}개 종목)")

    # C. FRED 거시지표
    fred_exist = [c for c in FRED_LEVEL_COLS if c in df.columns]
    for col in fred_exist:
        if col == "FRED_T10Y2Y":
            feat[col] = df[col]
            feat = add_lags(feat, col, LAG_MONTHS)
            feat[f"{col}_chg3"] = df[col].diff(3)
        else:
            yc = f"{col}_YoY"
            feat[yc] = yoy_pct(df[col])
            feat = add_lags(feat, yc, LAG_MONTHS)
            feat = add_ma(feat, yc, [3, 6])
            feat = add_momentum(feat, yc)
            feat = add_accel(feat, yc)
    print(f"  C. FRED 거시지표 피처 추가 ({len(fred_exist)}개 시리즈)")

    # D. ISM PMI 파생
    if "FRED_ISM_Mfg" in df.columns:
        ism = df["FRED_ISM_Mfg"]
        feat["ISM_above50"] = (ism > 50).astype(int)
        feat["ISM_mom3"]    = ism.diff(3)

    # E. 계절성
    feat["month_sin"] = np.sin(2 * np.pi * feat.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat.index.month / 12)

    # F. 타겟 (T+6 YoY%)
    for tgt in TARGETS:
        if tgt in df.columns:
            col_name = f"TARGET_{tgt}_YoY_T{TARGET_HORIZON}"
            feat[col_name] = yoy_pct(df[tgt]).shift(-TARGET_HORIZON)

    # H. Bear 선행 피처 (T10Y3M / ISRATIO / DFF)
    if "FRED_T10Y3M" in df.columns:
        t3m = df["FRED_T10Y3M"]
        feat["T10Y3M"] = t3m
        feat["T10Y3M_chg3"] = t3m.diff(3)
        feat["T10Y3M_chg6"] = t3m.diff(6)
        feat["T10Y3M_inverted"] = (t3m < 0).astype(int)
        inv = (t3m < 0).astype(int)
        groups = (inv != inv.shift()).cumsum()
        feat["T10Y3M_inv_streak"] = (inv.groupby(groups).cumcount() + 1) * inv
        feat = add_lags(feat, "T10Y3M", LAG_MONTHS)

    if "FRED_InvSales" in df.columns:
        inv_s = df["FRED_InvSales"]
        feat["InvSales"] = inv_s
        feat["InvSales_diff3"] = inv_s.diff(3)
        feat["InvSales_diff6"] = inv_s.diff(6)
        feat = add_lags(feat, "InvSales", LAG_MONTHS)

    if "FRED_FedFunds" in df.columns:
        dff = df["FRED_FedFunds"]
        feat["FedFunds"] = dff
        feat["FedFunds_diff6"]  = dff.diff(6)
        feat["FedFunds_diff12"] = dff.diff(12)
        feat = add_lags(feat, "FedFunds", LAG_MONTHS)

    n_bear = sum(1 for c in ["FRED_T10Y3M", "FRED_InvSales", "FRED_FedFunds"] if c in df.columns)
    print(f"  H. Bear 선행 피처 추가 ({n_bear}개 시리즈)")

    # G. NaN 처리
    target_cols = [c for c in feat.columns if c.startswith("TARGET_")]
    feat_clean = feat.dropna(subset=[target_cols[0]]) if target_cols else feat
    feat_clean = feat_clean.ffill().dropna(axis=1, thresh=int(len(feat_clean) * 0.5))

    n_feat = len([c for c in feat_clean.columns if not c.startswith("TARGET_")])
    print(f"\n  ▶ 최종 피처셋: {feat_clean.shape[0]}개월 × {n_feat}개 피처 + {len(target_cols)}개 타겟")
    return feat_clean


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Step 1  데이터 수집 + 피처 엔지니어링")
    print("=" * 64)

    # 데이터 수집
    df_wsts  = load_wsts()
    df_wsts.to_csv(os.path.join(DATA_DIR, "wsts_monthly.csv"))

    df_fred  = fetch_fred()
    df_eq    = fetch_equity()
    df_merged = merge_all(df_wsts, df_fred, df_eq)
    df_merged.to_csv(MERGED_PATH)
    print(f"  → 병합 데이터 저장: {MERGED_PATH}")

    # 피처 엔지니어링
    feat = build_features(df_merged)
    feat.to_csv(FEATURES_PATH)
    print(f"  → 피처셋 저장: {FEATURES_PATH}")
    print(f"\n  기간: {feat.index.min().date()} ~ {feat.index.max().date()}")
    print("  Step 1 완료.")


if __name__ == "__main__":
    main()
