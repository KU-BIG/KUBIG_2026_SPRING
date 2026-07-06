"""FWI (Fire Weather Index) - CFS 표준 공식 직접 구현"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUT_DIR


def _ffmc(temp, rh, ws, rain, ffmc0):
    mo = 147.2 * (101 - ffmc0) / (59.5 + ffmc0)
    if rain > 0.5:
        rf = rain - 0.5
        if mo <= 150:
            mr = mo + 42.5 * rf * np.exp(-100/(251-mo)) * (1 - np.exp(-6.93/rf))
        else:
            mr = (mo + 42.5 * rf * np.exp(-100/(251-mo)) * (1 - np.exp(-6.93/rf))
                  + 0.0015 * (mo-150)**2 * np.sqrt(rf))
        mo = min(mr, 250.0)
    ed = 0.942*rh**0.679 + 11*np.exp((rh-100)/10) + 0.18*(21.1-temp)*(1-np.exp(-0.115*rh))
    ew = 0.618*rh**0.753 + 10*np.exp((rh-100)/10) + 0.18*(21.1-temp)*(1-np.exp(-0.115*rh))
    if mo > ed:
        ko = 0.424*(1-(rh/100)**1.7) + 0.0694*np.sqrt(ws)*(1-(rh/100)**8)
        kd = ko * 0.581 * np.exp(0.0365*temp)
        m = ed + (mo-ed) * 10**(-kd)
    elif mo < ew:
        kl = 0.424*(1-((100-rh)/100)**1.7) + 0.0694*np.sqrt(ws)*(1-((100-rh)/100)**8)
        kw = kl * 0.581 * np.exp(0.0365*temp)
        m = ew - (ew-mo) * 10**(-kw)
    else:
        m = mo
    return 59.5 * (250-m) / (147.2+m)


def _dmc(temp, rh, rain, dmc0, month):
    pr = dmc0
    if rain > 1.5:
        re = 0.92*rain - 1.27
        mo = 20 + np.exp(5.6348 - dmc0/43.43)
        b = (100/(0.5+0.3*dmc0) if dmc0 <= 33
             else (14-1.3*np.log(dmc0) if dmc0 <= 65
                   else 6.2*np.log(dmc0)-17.2))
        mr = mo + 1000*re / (48.77 + b*re)
        pr = max(244.72 - 43.43*np.log(mr-20), 0)
    le = [6.5,7.5,9.0,12.8,13.9,13.9,12.4,10.9,9.4,8.0,7.0,6.0][month-1]
    k = max(1.894*(temp+1.1)*(100-rh)*le*1e-6, 0) if temp > -1.1 else 0
    return pr + 100*k


def _dc(temp, rain, dc0, month):
    dr = dc0
    if rain > 2.8:
        rw = 0.83*rain - 1.27
        Qo = 800*np.exp(-dc0/400)
        Qr = Qo + 3.937*rw
        dr = max(400*np.log(800/Qr), 0)
    lf = [-1.6,-1.6,-1.6,0.9,3.8,5.8,6.4,5.0,2.4,0.4,-1.6,-1.6][month-1]
    v = max(0.36*(temp+2.8)+lf, 0) if temp > -2.8 else 0
    return dr + 0.5*v


def _isi(ws, ffmc):
    fm = 147.2*(101-ffmc)/(59.5+ffmc)
    ff = 91.9*np.exp(-0.1386*fm)*(1+fm**5.31/49300000)
    return 0.208 * np.exp(0.05039*ws) * ff


def _bui(dmc, dc):
    denom = dmc + 0.4 * dc
    if denom == 0:
        return 0.0
    if dmc <= 0.4 * dc:
        return 0.8 * dmc * dc / denom
    return dmc - (1 - 0.8 * dc / denom) * (0.92 + (0.0114 * dmc) ** 1.7)


def _fwi_val(isi, bui):
    bui = max(bui, 0.0)
    bb = (0.1 * isi * (0.626 * bui ** 0.809 + 2) if bui <= 80
          else 0.1 * isi * (1000 / (25 + 108.64 * np.exp(-0.023 * bui))))
    bb = max(bb, 0.0)
    return bb ** 1.4641 if bb > 1 else bb


def calc_station(df_stn):
    df_stn = df_stn.sort_values("일시").reset_index(drop=True)
    ffmc, dmc, dc = 85.0, 6.0, 15.0
    rows = []
    for _, r in df_stn.iterrows():
        t  = float(r.get("최고기온(°C)", 15))
        rh = max(float(r.get("최소 상대습도(%)", 50)), 1.0)
        ws = float(r.get("최대 순간 풍속(m/s)", 0)) * 3.6
        rain = float(r.get("_rain_prev", 0))
        mo = r["일시"].month
        ffmc = _ffmc(t, rh, ws, rain, ffmc)
        dmc  = _dmc(t, rh, rain, dmc, mo)
        dc   = _dc(t, rain, dc, mo)
        isi  = _isi(ws, ffmc)
        bui  = _bui(dmc, dc)
        fwi  = _fwi_val(isi, bui)
        rows.append({"일시": r["일시"], "FFMC": ffmc, "DMC": dmc,
                     "DC": dc, "ISI": isi, "BUI": bui, "FWI": fwi})
    return pd.DataFrame(rows)


def run(aws_df):
    print("[FWI] 전 기간 연속 계산...")
    aws_df = aws_df.copy()
    aws_df["_rain_prev"] = aws_df.groupby("지점")["일강수량(mm)"].shift(1).fillna(0)
    stns = aws_df["지점"].unique()
    parts = []
    for i, stn in enumerate(stns, 1):
        sub = aws_df[aws_df["지점"] == stn]
        fwi_df = calc_station(sub)
        fwi_df["지점"] = stn
        parts.append(fwi_df)
        if i % 20 == 0:
            print(f"  {i}/{len(stns)} 지점...")
    fwi_all = pd.concat(parts, ignore_index=True)
    aws_df = aws_df.merge(fwi_all, on=["지점", "일시"], how="left")
    aws_df.drop(columns=["_rain_prev"], inplace=True)
    for col in ["FFMC","DMC","DC","ISI","BUI","FWI"]:
        aws_df[col] = aws_df[col].clip(lower=0)
    out = OUT_DIR / "aws_fwi.parquet"
    aws_df.to_parquet(out, index=False)
    print(f"[FWI] 저장: {out}  ({len(aws_df):,}행)")
    return aws_df


if __name__ == "__main__":
    aws = pd.read_parquet(OUT_DIR / "aws_processed.parquet")
    run(aws)
