"""
app.py — E2E 반도체 사이클 → SK하이닉스 수익률 예측 대시보드 (Streamlit)
================================================================================
Stage 1 (반도체 출하량 YoY 예측) → Stage 2 (SK하이닉스 6개월 수익률 방향 예측)
2단계 파이프라인의 학습 결과를 발표용으로 시각화한다.
"""

import os
import pickle
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

KST = timezone(timedelta(hours=9))

# ── 페이지 설정 (반드시 첫 Streamlit 호출) ──────────────────────────
st.set_page_config(
    page_title="반도체 사이클 → SK하이닉스 수익률 예측",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────────────────────────

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

W_BULL_CORRECT, W_BULL_WRONG = 1.0, 2.0
W_BEAR_CORRECT, W_BEAR_WRONG = 1.5, 3.0
BEAR_SAMPLE_W = 2.0

# ── 컬러 팔레트 ───────────────────────────────────────────────────
CLR_BLUE  = "#2a78d6"
CLR_TEAL  = "#1D9E75"
CLR_RED   = "#E24B4A"
CLR_AMBER = "#EF9F27"
CLR_GRAY  = "#888780"

BG_BLUE  = "#e6f1fb"
BG_GREEN = "#eaf3de"
BG_TEAL  = "#e1f5ee"
BG_RED   = "#fcebeb"
BG_AMBER = "#faeeda"

STAGE1 = {
    "name": "Stage 1",
    "title": "반도체 출하량 YoY 예측",
    "features_path": os.path.join(APP_ROOT, "stage1/outputs/data/features_dataset.csv"),
    "model_path":    os.path.join(APP_ROOT, "stage1/outputs/models/best_xgboost_final.pkl"),
    "target":        "TARGET_Worldwide_YoY_T6",
    "test_eval":     24,
    "value_label":   "예측 YoY",
    "freq_label":    "개월",
}
STAGE2 = {
    "name": "Stage 2",
    "title": "SK하이닉스 6개월 수익률 방향 예측",
    "features_path": os.path.join(APP_ROOT, "stage2/outputs/data/stage2_features.csv"),
    "model_path":    os.path.join(APP_ROOT, "stage2/outputs/models/skh_xgb_final.pkl"),
    "target":        "TARGET_SKH_6M_RET",
    "test_eval":     12,
    "value_label":   "예측 수익률",
    "freq_label":    "분기",
}

STAGE1_PRED_PATH = os.path.join(APP_ROOT, "stage2/outputs/data/stage1_predictions.csv")
BRIDGE_COL = "v2_pred_ww_yoy"


# ──────────────────────────────────────────────────────────────────
# 1. 데이터 / 모델 로딩 (캐싱)
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ──────────────────────────────────────────────────────────────────
# 3. 지표 계산 (hold-out 평가 재현)
# ──────────────────────────────────────────────────────────────────

def _safe_rmse(y_true, y_pred, mask):
    if not mask.any():
        return None
    err = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, with_ic: bool = False) -> dict:
    bull    = y_true > 0
    bear    = ~bull
    correct = (y_true > 0) == (y_pred > 0)
    w = np.where(bull & correct,  W_BULL_CORRECT,
        np.where(bull & ~correct, W_BULL_WRONG,
        np.where(bear & correct,  W_BEAR_CORRECT, W_BEAR_WRONG)))

    metrics = {
        "rmse":      float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "rmse_bull": _safe_rmse(y_true, y_pred, bull),
        "rmse_bear": _safe_rmse(y_true, y_pred, bear),
        "dir_acc":   float(correct.mean() * 100),
        "dir_bull":  float(correct[bull].mean() * 100) if bull.any() else None,
        "dir_bear":  float(correct[bear].mean() * 100) if bear.any() else None,
        "asym_loss": float(np.sqrt((w * (y_true - y_pred) ** 2).sum() / w.sum())),
    }
    if with_ic:
        ic = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        metrics["ic"] = float(ic) if pd.notna(ic) else None
    return metrics


@st.cache_data(show_spinner="모델 성능을 평가하는 중...")
def evaluate_stage(features_path: str, model_path: str, target: str,
                   test_eval: int, with_ic: bool = False):
    import xgboost as xgb

    bundle  = load_model(model_path)
    model   = bundle["model"]
    feats   = bundle["feature_names"]
    params  = model.get_params()

    df = load_csv(features_path)
    use_feats = [f for f in feats if f in df.columns]
    df_clean  = df.dropna(subset=[target])
    X = df_clean[use_feats].ffill().fillna(0)
    y = df_clean[target]

    split = len(X) - test_eval
    X_tune, y_tune = X.iloc[:split], y.iloc[:split]
    X_ho,   y_ho   = X.iloc[split:], y.iloc[split:]

    w_tune = np.where(y_tune.values > 0, 1.0, BEAR_SAMPLE_W)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tune, y_tune, sample_weight=w_tune)
    preds = m.predict(X_ho)

    metrics = compute_metrics(y_ho.values, preds, with_ic=with_ic)
    metrics["period"] = f"{y_ho.index[0].date()} ~ {y_ho.index[-1].date()}"
    metrics["n_holdout"] = len(y_ho)
    metrics["n_features"] = len(use_feats)

    out = pd.DataFrame({"실제값": y_ho.values, "예측값": preds}, index=y_ho.index)
    return metrics, out


@st.cache_data(show_spinner="SHAP 피처 중요도 계산 중...")
def compute_shap_importance(model_path: str, features_path: str, target: str,
                            top_n: int = 10) -> pd.DataFrame:
    import shap

    bundle = load_model(model_path)
    model  = bundle["model"]
    feats  = bundle["feature_names"]

    df = load_csv(features_path)
    use_feats = [f for f in feats if f in df.columns]
    df_clean  = df.dropna(subset=[target])
    X = df_clean[use_feats].ffill().fillna(0)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs    = np.abs(shap_values).mean(axis=0)

    s = (pd.Series(mean_abs, index=use_feats)
         .sort_values(ascending=False)
         .head(top_n))
    return s.rename("평균 |SHAP|").to_frame()


@st.cache_data(ttl=3600, show_spinner="시장 신호(yfinance) 수집 중...")
def get_market_momentum() -> dict:
    import yfinance as yf

    result = {}
    for label, ticker in [("KOSPI", "^KS11"), ("SOX", "^SOX")]:
        try:
            data  = yf.download(ticker, period="3mo", interval="1d",
                                progress=False, auto_adjust=True)
            close = np.asarray(data["Close"]).reshape(-1)
            close = close[~np.isnan(close)]
            if len(close) >= 2:
                result[label] = float(close[-1] / close[0] - 1.0) * 100
            else:
                result[label] = None
        except Exception:
            result[label] = None
    return result


# ──────────────────────────────────────────────────────────────────
# 4. UI 헬퍼
# ──────────────────────────────────────────────────────────────────

def _fmt(v, pct=False):
    if v is None:
        return "N/A"
    return f"{v:.1f}%" if pct else f"{v:.3f}"


def _fmt_bear(v):
    if v is None:
        return "해당 기간 하락 구간 없음"
    return f"{v:.1f}%"


def _pill(text: str, bg: str, fg: str) -> str:
    return (
        f"<span style='background:{bg};color:{fg};font-size:11px;font-weight:500;"
        f"padding:3px 10px;border-radius:20px;white-space:nowrap'>{text}</span>"
    )


def _chart_legend(*items) -> str:
    badges = " ".join(_pill(label, bg, fg) for label, bg, fg in items)
    return f"<div style='display:flex;gap:8px;margin-top:10px;flex-wrap:wrap'>{badges}</div>"


def _inject_styles():
    st.markdown("""
<style>
.signal-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 0; border-bottom: 0.5px solid rgba(136,135,128,0.25);
}
.signal-row:last-child { border-bottom: none; }
.signal-label { font-size: 13px; color: var(--text-secondary, #555); }
.signal-val   { font-size: 13px; font-weight: 500; }
.signal-val.up  { color: #1D9E75; }
.signal-val.dn  { color: #E24B4A; }
.signal-val.neu { color: #EF9F27; }

.cb-label { display:flex; justify-content:space-between; font-size:12px;
            color:#888; margin-bottom:5px; }
.cb-track { height:8px; border-radius:4px; background:rgba(136,135,128,0.2); overflow:hidden; }
.cb-fill  { height:100%; border-radius:4px; }

.caution-box { background:#faeeda; border-radius:10px; padding:14px; margin-top:12px; }
.caution-box .c-title { font-size:12px; font-weight:500; color:#854F0B; margin-bottom:6px; }
.caution-box .c-body  { font-size:12px; color:#633806; line-height:1.7; }

.expert-banner { background:#e6f1fb; border-radius:10px; padding:12px 16px;
                 margin-bottom:16px; border-left:3px solid #2a78d6; }
.eb-title { font-size:13px; font-weight:500; color:#0C447C; margin-bottom:2px; }
.eb-body  { font-size:12px; color:#185FA5; line-height:1.6; }

.kpi-card { background:var(--secondary-background-color,#f8f9fa);
            border-radius:10px; border:0.5px solid rgba(136,135,128,0.25);
            padding:14px 16px; }
.kpi-label { font-size:12px; color:#888; margin-bottom:4px; }
.kpi-value { font-size:22px; font-weight:500; color:var(--text-color,#111);
             margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)


def _expert_banner():
    st.markdown("""
<div class="expert-banner">
  <div class="eb-title">🔬 전문가 모드 켜짐</div>
  <div class="eb-body">판단 근거, 모델 수치, 주의사항을 상세하게 보여줘요.</div>
</div>
""", unsafe_allow_html=True)


def _confidence_bar(pct: float, label: str, color: str = CLR_BLUE):
    st.markdown(f"""
<div class="cb-label">
  <span>{label}</span>
  <span style="color:{color};font-weight:500">{pct:.0f}%</span>
</div>
<div class="cb-track">
  <div class="cb-fill" style="width:{pct:.0f}%;background:{color}"></div>
</div>
""", unsafe_allow_html=True)


def _signal_rows(rows: list):
    html = ""
    for label, val, direction in rows:
        cls = {"up": "up", "dn": "dn", "neu": "neu"}.get(direction, "")
        html += (
            f"<div class='signal-row'>"
            f"<span class='signal-label'>{label}</span>"
            f"<span class='signal-val {cls}'>{val}</span>"
            f"</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def _caution_box(text: str):
    st.markdown(f"""
<div class="caution-box">
  <div class="c-title">⚠️ 주의사항</div>
  <div class="c-body">{text}</div>
</div>
""", unsafe_allow_html=True)


def render_ribbon_chart(out_df: pd.DataFrame, rmse: float, height: int = 380):
    """Plotly 리본 차트: 80%/95% 신뢰구간 밴드 + 예측 파선 + 실제값 실선."""
    dates   = out_df.index.tolist()
    actual  = out_df["실제값"].tolist()
    pred    = out_df["예측값"].tolist()
    sigma   = rmse

    upper95 = [p + 1.96 * sigma for p in pred]
    lower95 = [p - 1.96 * sigma for p in pred]
    upper80 = [p + 1.28 * sigma for p in pred]
    lower80 = [p - 1.28 * sigma for p in pred]

    all_vals = actual + pred + upper95 + lower95
    y_min = min(all_vals)
    y_max = max(all_vals)
    pad   = (y_max - y_min) * 0.08
    y_lo  = y_min - pad
    y_hi  = y_max + pad

    fig = go.Figure()

    # ── 상승/하락 영역 배경 shading ──
    fig.add_shape(type="rect", xref="paper", yref="y",
        x0=0, x1=1, y0=0, y1=y_hi,
        fillcolor="rgba(29,158,117,0.04)", line_width=0, layer="below")
    fig.add_shape(type="rect", xref="paper", yref="y",
        x0=0, x1=1, y0=y_lo, y1=0,
        fillcolor="rgba(226,75,74,0.04)", line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper95 + lower95[::-1],
        fill="toself", fillcolor="rgba(150,150,150,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper80 + lower80[::-1],
        fill="toself", fillcolor="rgba(120,120,120,0.22)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="80% CI",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pred,
        line=dict(color=CLR_BLUE, width=2, dash="dash"),
        showlegend=False, name="예측 중앙값",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        line=dict(color=CLR_TEAL, width=2),
        mode="lines+markers", marker=dict(size=5, color=CLR_TEAL),
        showlegend=False, name="실제값",
    ))

    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(
            range=[y_lo, y_hi],
            gridcolor="rgba(136,135,128,0.12)",
            zeroline=True,
            zerolinecolor="rgba(100,100,100,0.45)",
            zerolinewidth=1,
            tickfont=dict(size=11),
        ),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(_chart_legend(
        ("실제값", "#d4f5e7", "#0a6e48"),
        ("예측 중앙값", BG_BLUE, "#185FA5"),
        ("80% 신뢰구간", "#ebebeb", "#555"),
        ("95% 신뢰구간", "#f2f2f2", "#888"),
    ), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# 5. 공통 섹션 렌더러
# ──────────────────────────────────────────────────────────────────

def render_direction_headline(out_df: pd.DataFrame, value_label: str):
    pred = float(out_df["예측값"].iloc[-1])
    date = out_df.index[-1]
    up   = pred > 0

    emoji  = "📈" if up else "📉"
    label  = "상승" if up else "하락"
    color  = CLR_TEAL if up else CLR_RED
    bg     = BG_TEAL  if up else BG_RED

    target_date = date + pd.DateOffset(months=6)

    direction_pill = _pill("▲ 상승 전망" if up else "▼ 하락 전망", bg, color)
    ai_pill        = _pill("AI 예측", BG_BLUE, "#185FA5")

    st.markdown(
        f"<div style='text-align:center;padding:1.5rem;border-radius:16px;"
        f"background:{bg};margin:0.2rem 0 0.4rem 0;'>"
        f"<div style='font-size:3.4rem;font-weight:500;color:{color};line-height:1.15'>"
        f"{emoji} {label} 전망</div>"
        f"<div style='margin-top:10px;display:flex;gap:8px;justify-content:center'>"
        f"{ai_pill} {direction_pill}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"📅 **{date.strftime('%Y년 %m월')}까지의 데이터** 기준 → "
        f"**{target_date.strftime('%Y년 %m월')} 방향** 예측 · "
        f"{value_label} {pred:+.2f}%"
    )


def _kpi(label: str, value: str, pill_text: str = None,
         pill_bg: str = BG_BLUE, pill_fg: str = "#185FA5"):
    pill_html = _pill(pill_text, pill_bg, pill_fg) if pill_text else ""
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"{pill_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _pill_grade(v):
    """정확도 수치 → (pill_text, bg, fg) 튜플."""
    if v is None:
        return None
    if v >= 80:
        return f"{v:.0f}%", BG_GREEN, "#3B6D11"
    if v >= 60:
        return f"{v:.0f}%", BG_AMBER, "#854F0B"
    return f"{v:.0f}%", BG_RED, "#A32D2D"


def render_metric_cards(metrics: dict, with_ic: bool = False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p = _pill_grade(metrics["dir_acc"])
        _kpi("방향 정확도 (전체)", _fmt(metrics["dir_acc"], pct=True),
             p[0] if p else None, p[1] if p else BG_BLUE, p[2] if p else "#185FA5")
    with c2:
        p = _pill_grade(metrics.get("dir_bull"))
        _kpi("방향 정확도 (Bull)", _fmt(metrics.get("dir_bull"), pct=True),
             p[0] if p else None, p[1] if p else BG_BLUE, p[2] if p else "#185FA5")
    with c3:
        p = _pill_grade(metrics.get("dir_bear"))
        _kpi("방향 정확도 (Bear)", _fmt_bear(metrics.get("dir_bear")),
             p[0] if p else None, p[1] if p else BG_BLUE, p[2] if p else "#185FA5")
    with c4:
        _kpi("RMSE (전체)", _fmt(metrics["rmse"]), "오차 지표", BG_BLUE, "#185FA5")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        _kpi("RMSE (Bull)", _fmt(metrics.get("rmse_bull")))
    with c6:
        _kpi("RMSE (Bear)", _fmt(metrics.get("rmse_bear")))
    with c7:
        _kpi("Asymmetric Loss", _fmt(metrics.get("asym_loss")))
    with c8:
        if with_ic:
            _kpi("IC (Spearman)", _fmt(metrics.get("ic")))
        else:
            _kpi("Hold-out 구간", f"{metrics['n_holdout']}개")


def _build_feat_map() -> dict:
    m = {}

    # ── WSTS 지역별 파생 피처 (build_features() Section A 완전 열거) ──
    _regions = {
        "Americas": "미주", "Europe": "유럽", "Japan": "일본",
        "Asia_Pacific": "아태지역", "Worldwide": "전세계",
    }
    for r, ko in _regions.items():
        b = f"{r}_YoY"
        m[b]                       = f"{ko} 반도체 YoY"
        m[f"{b}_lag6"]             = f"{ko} 반도체 YoY (6개월 전)"
        m[f"{b}_lag12"]            = f"{ko} 반도체 YoY (12개월 전)"
        m[f"{b}_ma3"]              = f"{ko} 반도체 YoY (3개월 평균)"
        m[f"{b}_ma6"]              = f"{ko} 반도체 YoY (6개월 평균)"
        m[f"{b}_ma12"]             = f"{ko} 반도체 YoY (12개월 평균)"
        m[f"{b}_vol3"]             = f"{ko} 반도체 YoY (3개월 변동성)"
        m[f"{b}_vol6"]             = f"{ko} 반도체 YoY (6개월 변동성)"
        m[f"{b}_momentum_3_12"]    = f"{ko} 반도체 YoY 모멘텀"
        m[f"{b}_accel"]            = f"{ko} 반도체 YoY 가속도"
        m[f"{b}_vs_ma24"]          = f"{ko} 반도체 YoY (24개월 내 상대 위치)"
        m[f"wsts_{r}_YoY"]         = f"{ko} 반도체 매출 YoY"

    # ── 주식 수익률 피처 (Section B) ──
    _tickers = {
        "SOX": "반도체지수 SOX", "NVDA": "NVIDIA", "TSM": "TSMC",
        "ASML": "ASML", "Samsung": "삼성전자", "SKHynix": "SK하이닉스",
    }
    for t, ko in _tickers.items():
        b = f"Ret_{t}"
        m[b]            = f"{ko} 수익률"
        m[f"{b}_lag6"]  = f"{ko} 수익률 (6개월 전)"
        m[f"{b}_lag12"] = f"{ko} 수익률 (12개월 전)"
        m[f"{b}_ma3"]   = f"{ko} 수익률 (3개월 평균)"
        m[f"{b}_ma6"]   = f"{ko} 수익률 (6개월 평균)"
        m[f"{b}_vol3"]  = f"{ko} 수익률 (3개월 변동성)"
        m[f"{b}_vol6"]  = f"{ko} 수익률 (6개월 변동성)"
    m["Eq_AvgRet"]       = "반도체 기업 평균 수익률"
    m["Eq_AvgRet_lag6"]  = "반도체 기업 평균 수익률 (6개월 전)"
    m["Eq_AvgRet_lag12"] = "반도체 기업 평균 수익률 (12개월 전)"

    # ── FRED 거시지표 (Section C) ──
    _fred = {
        "FRED_SemiProd":  "반도체 생산지수 (미국)",
        "FRED_ISM_Mfg":   "ISM 제조업 지수",
        "FRED_IndProd":   "산업생산지수 (미국)",
        "FRED_PCE_Core":  "근원 PCE 물가",
        "FRED_MfgEmp":    "제조업 고용",
        "FRED_ConsSenti": "소비자 심리지수",
        "FRED_NewOrder":  "제조업 신규 주문",
        "FRED_InvSales":  "재고/매출 비율",
        "FRED_FedFunds":  "연방기금금리",
    }
    for key, ko in _fred.items():
        b = f"{key}_YoY"
        m[b]                    = f"{ko} YoY"
        m[f"{b}_lag6"]          = f"{ko} YoY (6개월 전)"
        m[f"{b}_lag12"]         = f"{ko} YoY (12개월 전)"
        m[f"{b}_ma3"]           = f"{ko} YoY (3개월 평균)"
        m[f"{b}_ma6"]           = f"{ko} YoY (6개월 평균)"
        m[f"{b}_momentum_3_12"] = f"{ko} YoY 모멘텀"
        m[f"{b}_accel"]         = f"{ko} YoY 가속도"
    # T10Y2Y: YoY 변환 없이 원본 사용
    m["FRED_T10Y2Y"]       = "장단기 금리차 (10년-2년)"
    m["FRED_T10Y2Y_lag6"]  = "장단기 금리차 (6개월 전)"
    m["FRED_T10Y2Y_lag12"] = "장단기 금리차 (12개월 전)"
    m["FRED_T10Y2Y_chg3"]  = "장단기 금리차 변화 (3개월)"

    # ── ISM 파생 (Section D) ──
    m["ISM_above50"] = "ISM 50 초과 여부 (제조업 확장)"
    m["ISM_mom3"]    = "ISM 3개월 모멘텀"

    # ── 계절성 (Section E) ──
    m["month_sin"] = "계절성 (사인)"
    m["month_cos"] = "계절성 (코사인)"

    # ── Bear 선행 피처 (Section H) ──
    m["T10Y3M"]              = "장단기 금리차 (10년-3개월)"
    m["T10Y3M_chg3"]         = "금리차 3개월 변화"
    m["T10Y3M_chg6"]         = "금리차 6개월 변화"
    m["T10Y3M_inverted"]     = "금리 역전 여부"
    m["T10Y3M_inv_streak"]   = "금리 역전 연속 기간"
    m["T10Y3M_lag6"]         = "장단기 금리차 (6개월 전)"
    m["T10Y3M_lag12"]        = "장단기 금리차 (12개월 전)"
    m["InvSales"]            = "재고/매출 비율"
    m["InvSales_diff3"]      = "재고/매출 변화 (3개월)"
    m["InvSales_diff6"]      = "재고/매출 변화 (6개월)"
    m["InvSales_lag6"]       = "재고/매출 비율 (6개월 전)"
    m["InvSales_lag12"]      = "재고/매출 비율 (12개월 전)"
    m["FedFunds"]            = "연방기금금리"
    m["FedFunds_diff6"]      = "금리 변화 (6개월)"
    m["FedFunds_diff12"]     = "금리 변화 (12개월)"
    m["FedFunds_lag6"]       = "연방기금금리 (6개월 전)"
    m["FedFunds_lag12"]      = "연방기금금리 (12개월 전)"

    # ── Bridge (Stage 1 → Stage 2) ──
    m["v2_pred_ww_yoy"]       = "AI 반도체 경기 예측 (1단계 출력)"
    m["v2_pred_vs_current"]   = "AI 예측 vs 현재 YoY 괴리"
    m["v2_pred_bull"]         = "AI 반도체 경기 상승 신호"

    # ── Stage 2 전용: SK하이닉스 기술적 지표 ──
    m["SKH_price_obs"]     = "SK하이닉스 관찰 주가"
    m["SKH_log_price_obs"] = "SK하이닉스 관찰 주가 (로그)"
    m["SKH_ret_1m"]        = "SK하이닉스 수익률 (1개월)"
    m["SKH_ret_3m"]        = "SK하이닉스 수익률 (3개월)"
    m["SKH_ret_6m"]        = "SK하이닉스 수익률 (6개월)"
    m["SKH_ret_12m"]       = "SK하이닉스 수익률 (12개월)"
    m["SKH_vol_60d"]       = "SK하이닉스 변동성 (60일)"
    m["SKH_RSI_14"]        = "SK하이닉스 RSI (14일)"
    m["SKH_vs_ma60"]       = "SK하이닉스 vs MA60"
    m["SKH_vs_ma120"]      = "SK하이닉스 vs MA120"
    m["SKH_52w_pct"]       = "SK하이닉스 52주 고저 위치"

    # ── Stage 2 전용: 시장 센티먼트 ──
    m["VIX_level"]      = "VIX 공포지수"
    m["VIX_chg_1m"]     = "VIX 변화 (1개월)"
    m["SOX_vs_SPX_3m"]  = "SOX vs S&P500 상대 수익률 (3개월)"

    _s2_tickers = {
        "SOX": "반도체지수 SOX", "NVDA": "NVIDIA", "TSM": "TSMC",
        "ASML": "ASML", "Samsung": "삼성전자", "SPX": "S&P500",
    }
    for t, ko in _s2_tickers.items():
        for mo in [1, 3, 6]:
            m[f"{t}_ret_{mo}m"] = f"{ko} 수익률 ({mo}개월)"

    # ── Stage 2 전용: WSTS (분기 기준) ──
    m["WSTS_WW_YoY"]          = "전세계 반도체 매출 YoY"
    m["WSTS_WW_YoY_ma3"]      = "전세계 반도체 매출 YoY (3분기 평균)"
    m["WSTS_WW_YoY_ma6"]      = "전세계 반도체 매출 YoY (6분기 평균)"
    m["WSTS_WW_YoY_mom_3_12"] = "전세계 반도체 YoY 모멘텀"
    m["WSTS_WW_cycle_pos"]    = "전세계 반도체 사이클 위치"
    m["WSTS_AP_YoY"]          = "아태지역 반도체 매출 YoY"
    m["WSTS_AP_YoY_ma3"]      = "아태지역 반도체 매출 YoY (3분기 평균)"

    # ── Stage 2 전용: FRED (분기 기준) ──
    m["T10Y2Y_chg3"]    = "장단기 금리차 변화 (3분기)"
    m["FedFunds_chg3"]  = "연방기금금리 변화 (3분기)"
    m["T10Y3M_chg3"]    = "장단기 금리차 변화 (3분기, 10년-3개월)"
    m["IndProd_YoY"]    = "산업생산지수 YoY"
    m["PCE_Core_YoY"]   = "근원 PCE 물가 YoY"
    m["ConsSenti"]      = "소비자 심리지수"
    m["ConsSenti_chg3"] = "소비자 심리지수 변화 (3분기)"

    # ── Stage 2 전용: 환율·원자재 ──
    m["Oil_ret_3m"]    = "WTI 유가 수익률 (3개월)"
    m["Oil_ret_6m"]    = "WTI 유가 수익률 (6개월)"
    m["USDKRW_ret_3m"] = "USD/KRW 환율 변화 (3개월)"
    m["USDKRW_ret_6m"] = "USD/KRW 환율 변화 (6개월)"

    # ── Stage 2 전용: 달력·사이클 ──
    m["earnings_quarter"] = "실적 발표 분기"
    m["quarter_sin"]      = "분기 계절성 (사인)"
    m["quarter_cos"]      = "분기 계절성 (코사인)"
    m["supercycle_pos"]   = "반도체 4년 슈퍼사이클 위치"
    m["years_since_2000"] = "2000년 이후 경과 연수"

    return m


_FEAT_MAP       = _build_feat_map()
_FEAT_MAP_LOWER = {k.lower(): v for k, v in _FEAT_MAP.items()}


def _translate_feat(name: str) -> str:
    return _FEAT_MAP.get(name) or _FEAT_MAP_LOWER.get(name.lower(), name)


def render_shap_section(cfg: dict):
    st.markdown("#### 🔬 예측에 영향을 준 주요 지표 (상위 10)")
    st.caption("막대가 길수록 해당 지표가 이번 예측에 더 크게 영향을 미쳤어요.")
    try:
        shap_df = compute_shap_importance(cfg["model_path"], cfg["features_path"], cfg["target"])
        labels = [_translate_feat(n) for n in shap_df.index.tolist()]
        fig = go.Figure(go.Bar(
            x=shap_df["평균 |SHAP|"].values,
            y=labels,
            orientation="h",
            marker_color=CLR_BLUE,
        ))
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=4, r=16, t=8, b=0),
            yaxis=dict(
                autorange="reversed",
                automargin=True,
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(size=12),
            ),
            xaxis=dict(gridcolor="rgba(136,135,128,0.15)", title="영향도"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP 계산을 수행하지 못했습니다: {e}")


def _confidence_level(dir_acc: float):
    """방향 정확도 → (한글 레벨, 전경색, 배경색)."""
    if dir_acc >= 75:
        return "높음", CLR_TEAL, BG_TEAL
    if dir_acc >= 60:
        return "보통", CLR_AMBER, BG_AMBER
    return "낮음", CLR_RED, BG_RED


def render_detail_sections(metrics: dict, out_df: pd.DataFrame,
                           stage: str, expert_mode: bool):
    """📊 한 줄 요약 + 🔍 세부 분석 아코디언 + 🎯 신뢰도 바."""
    dir_acc  = metrics.get("dir_acc", 0)
    dir_bear = metrics.get("dir_bear")
    rmse     = metrics.get("rmse", 0)
    asym     = metrics.get("asym_loss", 0)
    n_ho     = metrics.get("n_holdout", 0)
    pred_last = float(out_df["예측값"].iloc[-1])
    is_up     = pred_last > 0

    if not expert_mode:
        # ── 📊 한 줄 요약 (비전문가 전용) ──
        st.markdown("#### 📊 한 줄 요약")
        if is_up:
            st.markdown(
                "반도체 사이클 지표가 상승 구간을 가리키고 있어요 📈 "
                "**→ HBM 수요**가 시그널을 이끌고 있어요."
            )
        else:
            st.markdown(
                "현재 사이클 지표는 하락 구간을 시사하고 있어요 📉 "
                "**→ 재고 조정 국면** 에 주의가 필요해요."
            )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("방향 정확도", _fmt(dir_acc, pct=True))
        with c2:
            st.metric("오차 (RMSE)", f"{rmse:.3f}")
        with c3:
            if dir_bear is not None:
                st.metric("Bear 정확도", _fmt(dir_bear, pct=True))
        st.markdown("---")

    if not expert_mode:
        st.caption(
            "이 예측은 참고용이에요. "
            "실제 투자 결정에는 다양한 요소를 종합적으로 고려해주세요."
        )
    else:
        # ── 전문가: 리스크 비토글 → 예측수치 토글 ──
        st.markdown("#### 🔍 세부 분석")

        bear_dir = dir_bear or 0
        _signal_rows([
            ("Bear DirAcc 안정성", _fmt_bear(dir_bear),
             "dn" if bear_dir < 60 else "up"),
            ("RMSE 대비 예측 신뢰", f"{rmse:.2f}", "neu"),
        ])
        _caution_box(
            "이 예측은 과거 데이터 패턴 기반의 통계 모델 출력값입니다. "
            "규제 리스크, 지정학적 이벤트, 기업 내부 정보 등 구조적 변화는 "
            "반영되지 않습니다. 투자 결정 시 이 수치만 단독으로 활용하지 마세요."
        )

        with st.expander("📈 예측 vs 실제 흐름 (수치)", expanded=False):
            st.dataframe(out_df.style.format("{:.2f}"), use_container_width=True)


def render_confusion(df: pd.DataFrame):
    yt, yp = df["실제값"].values, df["예측값"].values
    tp = int(((yt > 0) & (yp > 0)).sum())
    fp = int(((yt <= 0) & (yp > 0)).sum())
    fn = int(((yt > 0) & (yp <= 0)).sum())
    tn = int(((yt <= 0) & (yp <= 0)).sum())
    cm = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["실제 상승", "실제 하락"],
        columns=["예측 상승", "예측 하락"],
    )
    st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)


def render_hit_history(out_df: pd.DataFrame, freq_label: str, expert_mode: bool = False):
    st.markdown("#### 🎯 과거 적중 히스토리 (Hold-out)")
    d = out_df.copy()
    d["적중"] = (d["실제값"] > 0) == (d["예측값"] > 0)
    n, hit = len(d), int(d["적중"].sum())
    acc = d["적중"].mean() * 100 if n else 0.0

    cA, cB = st.columns([1, 3])
    with cA:
        st.metric("적중률", f"{acc:.1f}%", f"{hit}/{n} {freq_label} 적중")
    with cB:
        timeline = " ".join("✅" if v else "❌" for v in d["적중"])
        st.markdown("**적중 타임라인** (왼쪽=과거 → 오른쪽=최근)")
        st.markdown(
            f"<div style='font-size:1.5rem;letter-spacing:2px'>{timeline}</div>",
            unsafe_allow_html=True,
        )

    if expert_mode:
        table = pd.DataFrame({
            "예측 방향": ["📈 상승" if v > 0 else "📉 하락" for v in d["예측값"]],
            "실제 방향": ["📈 상승" if v > 0 else "📉 하락" for v in d["실제값"]],
            "예측값": [f"{v:+.2f}%" for v in d["예측값"]],
            "실제값": [f"{v:+.2f}%" for v in d["실제값"]],
            "결과": ["✅" if v else "❌" for v in d["적중"]],
        }, index=d.index.strftime("%Y-%m"))
        st.dataframe(table, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# 6. Stage별 화면
# ──────────────────────────────────────────────────────────────────

def view_stage1(expert_mode: bool = False):
    cfg = STAGE1
    st.header("📦 반도체 경기 예측 (6개월 뒤)")
    st.caption("전 세계 반도체 매출이 1년 전보다 얼마나 늘지 예측해요. SK하이닉스 전망의 출발점이에요.")

    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"], cfg["test_eval"]
        )
    except Exception as e:
        st.error(f"Stage 1 평가 중 오류가 발생했습니다: {e}")
        return

    render_direction_headline(df, cfg["value_label"])

    if expert_mode:
        st.markdown("#### 📊 모델 성능 분석")
        st.caption(f"평가 구간: {metrics['period']}  ·  피처 {metrics['n_features']}개")
        render_metric_cards(metrics)
        st.markdown(
            "**Asymmetric Loss** — 하락 구간에 더 높은 페널티를 부여한 가중 RMSE예요. "
            "Bear 오예측 시 3배 페널티가 적용돼 하락 경고를 놓치지 않도록 설계됐어요. "
            "일반 RMSE와 같은 수치라면 평가 구간에 하락 샘플이 없는 경우예요."
        )

    render_detail_sections(metrics, df, "stage1", expert_mode)

    with st.expander("🔬 SHAP 피처 중요도"):
        render_shap_section(cfg)

    with st.expander("📉 백테스트 결과 — 과거 예측이 얼마나 맞았나요?"):
        st.caption(f"모델이 학습에 쓰지 않은 구간({metrics['period']})에서 예측값과 실제값을 비교한 검증 차트예요. 현재 예측과는 별개예요.")
        render_ribbon_chart(df, metrics["rmse"])

    with st.expander("🎯 신뢰도 & 적중 히스토리"):
        _dir_acc  = metrics["dir_acc"]
        _dir_bear = metrics.get("dir_bear")
        _conf_fg  = CLR_TEAL if _dir_acc >= 75 else (CLR_AMBER if _dir_acc >= 60 else CLR_RED)
        _confidence_bar(_dir_acc, "방향 정확도 기반 신뢰도", _conf_fg)
        if _dir_bear is not None:
            _bear_fg = (CLR_TEAL if _dir_bear >= 60
                        else (CLR_AMBER if _dir_bear >= 40 else CLR_RED))
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            _confidence_bar(_dir_bear, "Bear 정확도 (하락 예측 신뢰도)", _bear_fg)
        st.divider()
        render_hit_history(df, cfg["freq_label"], expert_mode)


def view_stage2(expert_mode: bool = False):
    cfg = STAGE2
    st.header("📈 SK하이닉스 주가 전망 (6개월)")
    st.caption("반도체 경기 예측을 바탕으로 SK하이닉스 주가가 오를지 내릴지 판단해요.")

    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"],
            cfg["test_eval"], with_ic=True
        )
    except Exception as e:
        st.error(f"Stage 2 평가 중 오류가 발생했습니다: {e}")
        return

    render_direction_headline(df, cfg["value_label"])

    if expert_mode:
        st.markdown("#### 📊 모델 성능 분석")
        st.caption(f"평가 구간: {metrics['period']}  ·  피처 {metrics['n_features']}개")
        render_metric_cards(metrics, with_ic=True)
        st.markdown(
            "**Asymmetric Loss** — 하락 구간에 더 높은 페널티를 부여한 가중 RMSE예요. "
            "Bear 오예측 시 3배 페널티가 적용돼 하락 경고를 놓치지 않도록 설계됐어요. "
            "일반 RMSE와 같은 수치라면 평가 구간에 하락 샘플이 없는 경우예요.  \n"
            "**IC (Spearman)** — 예측 수익률 순위와 실제 수익률 순위가 얼마나 일치하는지 "
            "나타내요. 1에 가까울수록 크기 예측도 정확하고, 0이면 순위 예측력 없음이에요."
        )

    render_detail_sections(metrics, df, "stage2", expert_mode)

    with st.expander("🔬 SHAP 피처 중요도"):
        render_shap_section(cfg)

    with st.expander("📉 백테스트 결과 — 과거 예측이 얼마나 맞았나요?"):
        st.caption(f"모델이 학습에 쓰지 않은 구간({metrics['period']})에서 예측값과 실제값을 비교한 검증 차트예요. 현재 예측과는 별개예요.")
        render_ribbon_chart(df, metrics["rmse"], height=340)

    with st.expander("🎯 신뢰도 & 적중 히스토리"):
        _dir_acc  = metrics["dir_acc"]
        _dir_bear = metrics.get("dir_bear")
        _conf_fg  = CLR_TEAL if _dir_acc >= 75 else (CLR_AMBER if _dir_acc >= 60 else CLR_RED)
        _confidence_bar(_dir_acc, "방향 정확도 기반 신뢰도", _conf_fg)
        if _dir_bear is not None:
            _bear_fg = (CLR_TEAL if _dir_bear >= 60
                        else (CLR_AMBER if _dir_bear >= 40 else CLR_RED))
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            _confidence_bar(_dir_bear, "Bear 정확도 (하락 예측 신뢰도)", _bear_fg)
        st.divider()
        render_hit_history(df, cfg["freq_label"], expert_mode)


def _flow_box(title: str, subtitle: str, code: str = None):
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(f"**{subtitle}**")
        if code:
            st.caption(f"`{code}`")


def _flow_arrow():
    st.markdown(
        "<div style='text-align:center;font-size:2.6rem;margin-top:1.6rem'>➡️</div>",
        unsafe_allow_html=True,
    )


def render_market_signals():
    st.subheader("📊 현재 시장 분위기")
    st.caption("코스피·미국 반도체지수는 **실시간** (최대 1시간 자동 갱신) · AI 예측 신호는 학습 검증 기준이에요.")

    mom = get_market_momentum()
    kospi_mom = mom.get("KOSPI")
    sox_mom   = mom.get("SOX")

    model_up = None
    try:
        _, out2 = evaluate_stage(
            STAGE2["features_path"], STAGE2["model_path"],
            STAGE2["target"], STAGE2["test_eval"], with_ic=True
        )
        model_up = bool(out2["예측값"].iloc[-1] > 0)
    except Exception:
        pass

    def _mom_str(v):
        return "N/A" if v is None else f"{v:+.1f}%"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🇰🇷 코스피 (최근 3개월)", _mom_str(kospi_mom),
                  delta=None if kospi_mom is None else f"{kospi_mom:+.1f}%")
        st.caption("실시간 · 최대 1시간 전 데이터")
    with c2:
        st.metric("💽 미국 반도체지수 (최근 3개월)", _mom_str(sox_mom),
                  delta=None if sox_mom is None else f"{sox_mom:+.1f}%")
        st.caption("실시간 · 최대 1시간 전 데이터")

    votes = []
    if kospi_mom is not None:
        votes.append(kospi_mom > 0)
    if sox_mom is not None:
        votes.append(sox_mom > 0)
    if model_up is not None:
        votes.append(model_up)

    with c3:
        if not votes:
            st.metric("🧭 종합 신호", "N/A")
        else:
            pos = sum(votes)
            if pos > len(votes) / 2:
                st.metric("🧭 종합 신호", "📈 상승 우세",
                          delta=f"{pos}/{len(votes)} 신호 상승")
            elif pos < len(votes) / 2:
                st.metric("🧭 종합 신호", "📉 하락 우세",
                          delta=f"-{len(votes)-pos}/{len(votes)} 신호 하락")
            else:
                st.metric("🧭 종합 신호", "➖ 중립")
        st.caption("AI 신호는 최신 모델 기준")

    st.caption(
        "※ 종합 신호 = 코스피·미국 반도체지수의 3개월 흐름 + AI 최신 예측을 합친 다수결이에요. "
        "주가지수는 실시간(최대 1시간 단위 갱신), AI 신호는 최신 예측 기준이에요."
    )


def _plain_acc(dir_acc) -> str:
    n = round((dir_acc or 0) / 10)
    return f"과거 검증에서 **10번 중 약 {n}번** 방향을 맞혔어요."


def view_home(expert_mode: bool = False):
    """🏠 한눈에 보기 — 결론 · 시장 분위기 · 신뢰도를 한 페이지로."""
    try:
        m2, df2 = evaluate_stage(
            STAGE2["features_path"], STAGE2["model_path"],
            STAGE2["target"], STAGE2["test_eval"], with_ic=True,
        )
    except Exception as e:
        st.error(f"예측 결과를 불러오지 못했습니다: {e}")
        return

    up = float(df2["예측값"].iloc[-1]) > 0

    # ── 결론 (가장 중요) ──
    st.markdown("### 🔮 앞으로 6개월, SK하이닉스 주가는 오를까요?")
    render_direction_headline(df2, STAGE2["value_label"])

    takeaway = ("AI는 향후 6개월 SK하이닉스 주가가 **오를 가능성**이 높다고 봐요."
                if up else
                "AI는 향후 6개월 SK하이닉스 주가가 **내릴 가능성**이 높다고 봐요.")
    st.info(f"{takeaway}  \n{_plain_acc(m2['dir_acc'])}")

    # ── 예측 과정 (쉬운 3단계) ──
    st.markdown("#### 🧭 이렇게 예측해요")
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            st.markdown("##### 🌐 1. 반도체 경기")
            st.caption("전 세계 반도체가 6개월 뒤 얼마나 팔릴지 예측해요.")
    with s2:
        with st.container(border=True):
            st.markdown("##### 🔗 2. 신호 연결")
            st.caption("반도체 경기 예측을 SK하이닉스 분석에 연결해요.")
    with s3:
        with st.container(border=True):
            st.markdown("##### 📈 3. 주가 전망")
            st.caption("SK하이닉스 주가가 오를지 내릴지 최종 판단해요.")

    st.divider()
    render_market_signals()

    st.divider()
    with st.expander("🎯 이 예측, 얼마나 믿을 수 있나요?"):
        dir_acc  = m2.get("dir_acc", 0)
        dir_bear = m2.get("dir_bear")
        st.markdown(_plain_acc(dir_acc))
        conf_color = CLR_TEAL if dir_acc >= 75 else (CLR_AMBER if dir_acc >= 60 else CLR_RED)
        _confidence_bar(dir_acc, "전체 방향 정확도", conf_color)
        if dir_bear is not None:
            bear_color = (CLR_TEAL if dir_bear >= 60
                          else (CLR_AMBER if dir_bear >= 40 else CLR_RED))
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            _confidence_bar(dir_bear, "하락장에서의 정확도", bear_color)
        st.caption("'검증'은 모델이 학습에 쓰지 않은 최근 데이터로 시험 본 결과예요. "
                   "참고용이며 투자 권유가 아니에요.")

    st.caption("👈 왼쪽 메뉴에서 단계별 상세 분석과 차트를 볼 수 있어요.")


def view_e2e(expert_mode: bool = False):
    st.header("🔗 작동 원리")

    st.markdown("#### ① Stage 1 출력 시계열")
    st.caption(f"lookahead 없이 재학습한 6개월 선행 반도체 매출 YoY 예측값(`{BRIDGE_COL}`)")
    try:
        s1pred = load_csv(STAGE1_PRED_PATH)
        if BRIDGE_COL in s1pred.columns:
            s1_data = s1pred[[BRIDGE_COL]].dropna()
            fig = go.Figure(go.Scatter(
                x=s1_data.index, y=s1_data[BRIDGE_COL],
                line=dict(color=CLR_BLUE, width=2),
                mode="lines+markers", marker=dict(size=4),
            ))
            fig.update_layout(
                height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False, margin=dict(l=0, r=0, t=8, b=0),
                yaxis=dict(gridcolor="rgba(136,135,128,0.15)"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"`{BRIDGE_COL}` 컬럼을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"Stage 1 예측 데이터 로드 실패: {e}")

    st.divider()
    st.markdown("#### ② 두 모델 연결 확인")
    st.caption(
        "1단계(반도체 경기 예측) 모델의 출력값이 2단계(SK하이닉스 전망) 모델의 "
        "입력 피처로 전달돼야 두 단계가 올바르게 연결돼요. "
        "이 섹션은 그 연결이 정상적으로 이루어졌는지 확인해요."
    )
    try:
        s2feat = load_csv(STAGE2["features_path"])
        if BRIDGE_COL in s2feat.columns:
            st.success(
                "✅ 두 단계 정상 연결 — "
                "반도체 경기 예측값(`v2_pred_ww_yoy`)이 SK하이닉스 전망 모델의 "
                f"입력 피처로 포함되어 있어요. (전체 피처 {s2feat.shape[1]}개 중 하나)"
            )
        else:
            st.warning("⚠️ 연결 피처를 찾지 못했습니다. 파이프라인 재실행이 필요할 수 있어요.")
    except Exception as e:
        st.error(f"Stage 2 피처 데이터 로드 실패: {e}")

    st.divider()
    st.markdown("#### ③ 두 단계 성능 요약")
    rows = []
    for cfg, with_ic in [(STAGE1, False), (STAGE2, True)]:
        try:
            m, _ = evaluate_stage(
                cfg["features_path"], cfg["model_path"], cfg["target"],
                cfg["test_eval"], with_ic=with_ic
            )
            rows.append({
                "단계": f"{cfg['name']} · {cfg['title']}",
                "방향정확도(전체)": _fmt(m["dir_acc"], pct=True),
                "방향정확도(Bear)": _fmt_bear(m.get("dir_bear")),
                "RMSE": _fmt(m["rmse"]),
                "Asym Loss": _fmt(m["asym_loss"]),
                "IC": _fmt(m.get("ic")) if with_ic else "—",
            })
        except Exception as e:
            rows.append({"단계": cfg["name"], "방향정확도(전체)": f"오류: {e}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────
# 7. 메인
# ──────────────────────────────────────────────────────────────────

def main():
    _inject_styles()

    st.sidebar.title("📈 SK하이닉스 주가 전망")
    st.sidebar.caption("반도체 경기로 6개월 뒤 주가 방향을 예측해요")

    view = st.sidebar.radio(
        "메뉴",
        ["🏠 한눈에 보기", "A.  SK하이닉스 전망", "B.  반도체 경기", "작동 원리"],
        index=0,
    )

    st.sidebar.divider()
    expert_mode = st.sidebar.toggle("🔬 전문가 모드", value=False)
    st.sidebar.caption("SHAP·RMSE 등 전문 지표와 상세 수치를 함께 보여줘요.")

    st.sidebar.divider()
    with st.sidebar.expander("이 서비스는 어떻게 작동하나요?"):
        st.markdown(
            "**1. 데이터 수집**<br>"
            "전 세계 반도체 출하량(WSTS), 미국 경제지표(FRED), 주요 반도체 기업 주가를 자동으로 모읍니다.\n\n"
            "**2. 반도체 경기 예측 (B)**<br>"
            "수집한 데이터를 AI 모델에 넣어 6개월 뒤 반도체 시장이 성장할지 예측합니다.\n\n"
            "**3. SK하이닉스 주가 전망 (A)**<br>"
            "반도체 경기 예측 결과를 포함한 신호들로 SK하이닉스 주가가 6개월 뒤 오를지 내릴지 판단합니다.\n\n"
            "**4. 주기적 업데이트**<br>"
            "분기마다 (1·4·7·10월) 새 데이터로 모델을 다시 학습해 예측을 갱신합니다.",
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    st.sidebar.markdown(
        f"<div style='font-size:11px;color:#999;line-height:1.8'>"
        f"📡 <b>코스피·SOX</b>: 실시간 (1시간 갱신)<br>"
        f"🤖 <b>AI 예측 기준</b>: 최신 분기<br>"
        f"🕐 <b>페이지 로드</b>: {now_kst}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='font-size:11px;color:#aaa;line-height:1.7;text-align:center'>"
        "고려대학교 KUBIG<br>"
        "26학년도 1학기 컨퍼런스 스터디"
        "</div>",
        unsafe_allow_html=True,
    )

    if expert_mode:
        _expert_banner()

    st.title("반도체 사이클 기반 SK하이닉스 수익률 예측")

    if view.startswith("🏠"):
        view_home(expert_mode)
    elif view.startswith("A."):
        view_stage2(expert_mode)
    elif view.startswith("B."):
        view_stage1(expert_mode)
    else:
        view_e2e(expert_mode)


if __name__ == "__main__":
    main()
