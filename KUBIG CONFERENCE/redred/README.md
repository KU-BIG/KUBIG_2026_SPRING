# 📈 반도체 업황 기반 SK하이닉스 주가 예측

#### 🚦 Team 신호등바뀌었어redred : 김석우, 윤채영, 이지윤

---

## 목차
 
1. [프로젝트 개요](#프로젝트-개요)
2. [대시보드](#대시보드)
3. [2-Stage 모델별 결과 요약](#2-Stage-모델별-결과-요약)
4. [평가 설계 원칙 (Stage 2)](#평가-설계-원칙-stage-2)
5. [환경 설정](#환경-설정)
6. [실행 방법](#실행-방법)
7. [파이프라인 단계별 설명](#파이프라인-단계별-설명)
8. [디렉토리 구조](#디렉토리-구조)

---

## 프로젝트 개요

### 풀고자 하는 문제

주식 시장은 미래를 선반영한다. 투자자들은 지금 업황이 아니라 **"앞으로의 업황이 어떻게 될 것인가"** 를 보고 현재에 미리 사고판다. 특히 반도체 산업의 사이클에서 이러한 패턴을 확인할 수 있다. 반도체 공급망의 흐름은 `글로벌 업황 변동 → 반도체 기업 실적 변동 → 주가 변동` 순서를 따른다. 이 인과 구조를 모델에 직접 반영하면 단순 패턴 매칭보다 해석 가능한 예측이 가능하다.

> **예시** : 2023년 초, 반도체 재고 조정이 한창이라 실제 업황(WSTS YoY%)은 여전히 마이너스였다. 그런데 SK하이닉스 주가는 이미 2023년 1월부터 오르기 시작했다. 시장이 "재고 소진이 끝나면 하반기부터 업황이 회복될 것"을 미리 주가에 반영했기 때문이다. 즉, 업황 개선에 대한 기대가 이 6개월 구간 안에서 주가에 반영된 것이다.

즉, **업황이 실제로 좋아지는 시점보다 주가가 6개월 정도 먼저 움직이는 패턴** 을 활용한다면, **"앞으로 6개월 후 업황 방향을 미리 알 수 있다면, 지금부터 6개월간 주가가 오를지 내릴지 예측할 수 있다"** 는 가설이 성립한다. 따라서 본 프로젝트는 이를 2단계로 모델링한다.

```
[반도체 업황 선행 예측] ──────────────────→ [SK하이닉스 주가 방향 예측]
     Stage 1            Bridge 피처            Stage 2
  (Worldwide YoY%,   (v2_pred_ww_yoy)    (6개월 후 종가 수익률
   6개월 선행, 월간)                            방향, 분기)
```

> **왜 6개월 후 주가 수익률의 값 자체가 아닌, 변동 방향(Bull/Bear)을 예측하는가?**
> SK하이닉스 6개월 수익률은 분기에 따라 -150% ~ +180%에 달하는 극단적인 변동폭을 보인다.
> 이런 타겟에서 "정확히 몇 %"를 맞히는 것은 구조적으로 어렵고, 투자 의사결정에서도
> "오를 것인가 내릴 것인가"가 우선적인 정보다. 

---

## 대시보드

Streamlit Cloud에 배포된 공개 대시보드에서 예측 결과를 확인할 수 있습니다.

> 🔗 [대시보드 바로가기](https://kubig26-1conference-jtydr4ccfejqcsoz3wwsms.streamlit.app/)

### 대시보드 동작 방식

사용자가 대시보드에 접속하면, 사전에 파이프라인에 따라 학습·저장된 모델 파라미터(pkl)와 피처 데이터(csv)를 로드해 Stage 2 XGBoost 모델을 그 자리에서 재학습한 뒤, 가장 최근 Hold-out 구간을 예측해 "오를까 내릴까" 결과를 표시한다. 사이드바에서는 KOSPI·SOX 최근 3개월 등락률을 yfinance로 실시간 조회해 참고 시장 신호로 함께 보여준다.

### 자동 재학습 (GitHub Actions)

SK하이닉스 실적 발표일(1·4·7·10월 넷째 주 금요일)마다 자동으로 실행됩니다.

1. cron이 매월 22~28일 실행을 트리거하면, 당일이 KST 기준 금요일인지 확인 후 금요일이 아니면 즉시 종료 (넷째 주 금요일에만 재학습이 trigger되는 구조)
2. WSTS 사이트에서 최신 데이터(`wsts_historical.xlsx`) 자동 파싱·다운로드
3. Stage 1 → Stage 2 파이프라인 전체 순서대로 재학습
4. 모델(pkl), 피처 데이터(csv), 시각화(png), 지표(csv) 파일만 선택적으로 커밋·푸시
5. Streamlit Cloud가 새 커밋을 감지해 대시보드 자동 재배포

> GitHub Secrets에 `FRED_API_KEY`를 등록해야 합니다.
> Settings → Secrets and variables → Actions → New repository secret

---

## 2-Stage 모델별 결과 요약

### Stage 1️⃣ — 반도체 업황 YoY% 6개월 선행 예측 (월간)

**Stage 1의 Task**: WSTS 반도체 Worldwide 매출의 전년 대비 성장률(YoY%)이 6개월 후에 어떻게 될지를 예측한다. 양수(+)이면 업황 확장(Bull), 음수(-)이면 업황 수축(Bear)을 의미한다.

| 지표 | 값 |
|------|----|
| CV RMSE (5-fold) | **6.06** |
| CV DirAcc | **95.0%** (Bull 100.0% / Bear 87.1%) |
| CV AsymLoss | **6.01** |
| Hold-out RMSE (최근 24개월) | 17.41 |
| Hold-out DirAcc | 95.8% |
| 최종 선택 피처 수 | 30개 / 165개 |

> **AsymLoss**: Bear 국면(YoY% ≤ 0) 오예측에 ×3.0 페널티를 부여한 커스텀 손실함수.

### Stage 2️⃣ — SK하이닉스 6개월 후 종가 수익률 방향 예측 (분기)

**Stage 2의 Task**: 앞서 Stage 1에서 생성한 **6개월 후의 반도체 업황 예측값** 을 활용해서, 관찰일로부터 6개월 후 실적발표일까지의 SK하이닉스 종가 수익률의 변동 방향을 예측한다. 업황 개선 기대가 이 6개월 구간 안에서 주가에 선반영되는 패턴을 포착하는 것이 핵심이다. 양수(+)이면 상승(Bull), 음수(-)이면 하락(Bear) 신호다. 
`TARGET = (실적발표일 종가 / 관찰일 종가 − 1) × 100 (%)`

평가 환경: Tune 86분기 / Hold-out 20분기 (2021-01-28 ~ 2025-10-23)

| 지표 | 값 |
|------|--------------------------|
| CV RMSE (5-fold) | **18.974%** |
| CV DirAcc | **90.0%** (Bull 88.3% / Bear 100.0%) |
| CV AsymLoss | **19.320** |
| CV IC (Spearman) | **+0.600** |
| Hold-out RMSE | **62.389%** |
| Hold-out DirAcc (최근 20분기) | **60.0%** (Bull 91.7% / Bear 12.5%) |
| Hold-out IC | **+0.033** |
| 최적 recency_scale | 0.4087 |
| 최종 선택 피처 수 | 25개 |

---

## 평가 설계 원칙 (Stage 2)

### 왜 RMSE가 아닌 DirAcc를 최적화 기준으로 삼았는가

SK하이닉스 6개월 수익률은 분기에 따라 -150% ~ +180%에 달하는 극단적인 변동폭을 보인다. Hold-out 구간 타겟의 표준편차 자체가 약 60%p를 넘는 상황에서 RMSE를 직접 최소화하는 접근은 구조적으로 한계가 있으며, 특히 2025년의 AI 반도체 수요 급증으로 인한 +150%대 수익률은 어떤 과거 패턴으로도 정량 예측이 불가능한 외생적 충격이다.

실질적인 투자 의사결정 관점에서도 **"오를 것인가 내릴 것인가"** 가 우선적인 정보다. 따라서 DirAcc를 핵심 지표로, 방향 오예측에 대한 비대칭 페널티(Asymmetric Loss)를 학습 목적함수로 채택했다. 하락(Bear) 오예측에 가장 큰 페널티(×3.0)를 부여한 이유는, "상승을 놓치는 기회비용"보다 "하락을 못 보고 매수해 발생하는 원금 손실"이 더 치명적이기 때문이다.

```
Bull 정답: ×1.0  /  Bull 오답: ×2.0
Bear 정답: ×1.5  /  Bear 오답: ×3.0  ← 하락을 놓치는 것에 가장 큰 페널티
```

### 왜 Hold-out 구간을 20분기로 설정했는가

| TEST_EVAL_SIZE | 기간 | Bull | Bear | 판단 |
|---|---|---|---|---|
| 12분기 | 2023-01 ~ 2025-10 | 11 | 1 | Bear 표본 1개 → 지표가 0% or 100%만 가능, 통계적으로 무의미 |
| **20분기** | **2021-01 ~ 2025-10** | **12** | **8** | **Bull:Bear ≈ 60:40, 가장 균형적** ✅ |
| 24분기 | 2020-01 ~ 2025-10 | 15 | 9 | 학습 데이터 추가 축소, COVID 구간 포함 |

12분기 설정에서는 Bear 표본이 단 1개뿐이라 Bear DirAcc가 0% 또는 100%만 나오는 것을 실증적으로 확인했다. Bull/Bear 비율이 가장 균형적이면서 학습 데이터(86분기)를 충분히 확보할 수 있는 **20분기**를 최종 평가 기준으로 채택했다.

---

## 환경 설정

### 1. Python 버전

Python **3.9 이상**을 권장합니다.

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. FRED API 키 설정

```bash
# Mac / Linux
export FRED_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:FRED_API_KEY="your_api_key_here"
```

---

## 실행 방법

**Stage 1을 먼저 실행해야 합니다.** Stage 2는 Stage 1의 피처셋과 학습된 모델 파라미터를 입력으로 사용합니다.

### Stage 1

```bash
# 전체 파이프라인 한 번에 실행 (권장)
python stage1/pipeline.py

# 단계별 실행
python stage1/s1_data.py      # Step 1: 데이터 수집 + 피처 생성   
python stage1/s2_tune.py      # Step 2: RMSE 기준 Optuna 튜닝   
python stage1/s3_select.py    # Step 3: 피처 선택 (RFE)   
python stage1/s4_optimize.py  # Step 4: AsymLoss Bear 최적화
python stage1/s5_evaluate.py  # Step 5: 최종 평가 + 그래프 저장
```

### Stage 2

```bash
# 전체 파이프라인 한 번에 실행 (권장)
python stage2/pipeline.py

# 단계별 실행
python stage2/s1_dates.py       # Step 1: 분기 날짜 생성 + 타겟 수익률 산출 
python stage2/s2_data.py        # Step 2: A~E 피처 수집                  
python stage2/s3_stage1_feat.py # Step 3: Stage 1 Expanding Window 예측  
python stage2/s4_features.py    # Step 4: 피처 엔지니어링 + 피처 선택      
python stage2/s5_tune.py        # Step 5: XGBoost Optuna 튜닝         
python stage2/s6_evaluate.py    # Step 6: 최종 평가 + 그래프 저장       
```

> **주의**: 각 스텝은 이전 스텝의 출력 파일에 의존합니다. 반드시 순서대로 실행하세요.

---

## 파이프라인 단계별 설명

### Stage 1️⃣ 

#### Step 1 — 데이터 수집 + 피처 엔지니어링 (`s1_data.py`)

| 데이터 소스 | 내용 |
|------------|------|
| `wsts_historical.xlsx` | WSTS 월별 반도체 출하량 (Americas / Europe / Japan / Asia Pacific / Worldwide) |
| FRED API | 산업생산지수, 장단기 금리차(T10Y2Y·T10Y3M), 소비자심리지수, 신규수주, 제조업 고용, 연방기금금리 등 |
| yfinance | 필라델피아 반도체지수(SOX), NVDA, TSM, ASML, 삼성전자, SK하이닉스 주가 |

- 생성 피처: YoY% 기본값 / Lag(lag6·lag12) / 이동평균(ma3·ma6·ma12) / 변동성·모멘텀·가속도 / 사이클 위치 Percentile / Bear 선행 지표(T10Y3M 역전 여부, ISRATIO 등)
- 타겟: `TARGET_Worldwide_YoY_T6    = Worldwide 매출 YoY%  (shift(-6), 6개월 후)`

#### Step 2 — Optuna 하이퍼파라미터 튜닝 (`s2_tune.py`)

- 목적함수: TimeSeriesSplit 5-fold CV **RMSE 최소화**
- 탐색 파라미터: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`

#### Step 3 — 피처 선택 (`s3_select.py`)

1. **다중공선성 제거**: |Pearson r| ≥ 0.9인 피처 쌍에서 하나 제거
2. **SHAP 중요도 계산**: 전체 피처 중요도 순위 산출
3. **RFE 커브**: 피처 수 변화에 따른 CV AsymLoss 측정 후 최적 수 선택
4. 최종 선택: **30개** / 165개

#### Step 4 — Bear 최적화 (`s4_optimize.py`)

- 목적함수: **AsymLoss** (Bear 오예측 ×3.0 페널티) 최소화
- 선택된 30개 피처 + Bear 월 `sample_weight=2.0` 적용
- 출력: `stage1/outputs/models/best_xgboost_final.pkl` ← Stage 2 입력으로 활용됨

#### Step 5 — 최종 평가 + 시각화 (`s5_evaluate.py`)

- Hold-out 평가 (최근 24개월)
- 출력: 예측 타임라인, CV fold별 지표, Bear/Bull 개선 비교, SHAP 요약

---

### Stage 2️⃣

#### Step 1 — 분기 날짜 생성 + 타겟 수익률 산출 (`s1_dates.py`)

- SK하이닉스 **분기 실적발표일**: 매년 1·4·7·10월 넷째 주 목요일
- **관찰일**: 실적발표일 정확히 6개월 전 같은 요일
- 타겟: `TARGET_SKH_6M_RET = (P_earnings / P_obs − 1) × 100 (%)`

#### Step 2 — 피처 데이터 수집 (`s2_data.py`)

관찰일 기준 5개 피처 그룹 수집 (lookahead 없이, 관찰일 이전 데이터만 사용):

| 그룹 | 내용 |
|------|------|
| A. SK하이닉스 기술적 지표 | 가격, 1·3·6·12개월 수익률, 변동성(60d), RSI(14), MA 괴리율, 52주 고저 위치 |
| B. 시장 센티먼트 | VIX, SOX, NVDA, TSM, ASML, Samsung, S&P500의 1·3·6개월 수익률 |
| C. WSTS 역사 데이터 | Worldwide·Asia Pacific YoY%, 이동평균, 모멘텀, 사이클 위치 |
| D. FRED 거시지표 | 장단기 금리차(T10Y2Y·T10Y3M), 기준금리, 산업생산, PCE, 소비자심리 |
| E. 환율·원자재 | USD/KRW, WTI 유가의 3·6개월 수익률 |

#### Step 3 — Stage 1 Expanding Window Pseudo 예측 (`s3_stage1_feat.py`)

각 관찰일 기준으로 Stage 1 모델을 **expanding window** 방식으로 재훈련해 `v2_pred_ww_yoy`(관찰 시점 기준 6개월 후 업황 예측값)를 생성한다. 이 값이 Stage 2의 핵심 Bridge 피처로 활용된다.

> Expanding window를 쓰는 이유: 각 관찰 시점에서 미래 데이터를 전혀 사용하지 않고 OOS(out-of-sample) 예측값만 생성하기 위함이다. 단순히 전체 Stage 1 모델로 예측하면 lookahead가 발생한다.

#### Step 4 — 피처 엔지니어링 + 피처 선택 (`s4_features.py`)

- 달력·사이클 피처 추가: 실적 발표 분기, 반도체 슈퍼사이클 위치, 장기 추세 proxy
- `v2_pred_ww_yoy` 파생 피처: 예측 vs 현재 WSTS YoY% 괴리, Bull/Bear 신호
- **3단계 피처 선택**: NaN 비율 필터 → VIF 다중공선성 제거(임계 10) → XGBoost importance 상위 60% → RFE 최종 **25개**

#### Step 5 — XGBoost Optuna 튜닝 + Dynamic Sample Weight (`s5_tune.py`)

학습은 2단계로 나뉜다.

- **Phase A** — RMSE 최소화: XGBoost 하이퍼파라미터 초기 탐색 → `skh_xgb_tuned.pkl`
- **Phase B** — AsymLoss 최소화: 방향성 최적화 + Dynamic Sample Weight 적용 → `skh_xgb_final.pkl`
- CV 구조: TimeSeriesSplit 5-fold (test_size=4분기, min_train=20분기)

> **Dynamic Sample Weight** — 베이스라인 대비 최종 모델의 핵심 개선
>
> 베이스라인은 Bear 여부에 따른 고정 가중치만 사용한다(`Bull=1.0 / Bear=2.0`). 최종 모델은 여기에 시간 가중치(Recency Weight)를 결합해, 최근 분기일수록 더 높은 가중치로 학습하도록 설계했다. Hold-out 구간(2021~2025)이 AI 반도체 수요 급증으로 학습 구간 전체의 패턴과 분포가 달라지는 것(distribution shift)에 대응하기 위함이다.
>
> ```python
> def dynamic_weights(y, recency_scale):
>     recency_w = np.exp(np.linspace(0, recency_scale, len(y)))
>     recency_w = recency_w / recency_w.mean()          # 평균 1.0 정규화
>     bear_w    = np.where(y > 0, 1.0, BEAR_SAMPLE_W)  # 기존 Bear 가중치
>     return recency_w * bear_w
> ```
>
> `recency_scale`은 Optuna로 XGBoost 하이퍼파라미터와 함께 공동 탐색한다 (탐색 범위: 0.0~0.5, 최적값: 0.4087).
> `USE_DYNAMIC_WEIGHTS` 플래그(`True`/`False`)로 베이스라인과 최종 모델을 전환할 수 있다.

#### Step 6 — 최종 평가 + 시각화 (`s6_evaluate.py`)

평가 지표: RMSE (전체·Bull·Bear), DirAcc (전체·Bull·Bear), AsymLoss, IC (Spearman)

| 파일 | 내용 |
|------|------|
| `01_return_timeline.png` | 예측 vs 실제 수익률 전 기간 + Hold-out 확대 |
| `02_cv_metrics.png` | CV 평균 vs Hold-out 지표 바차트 |
| `03_direction_analysis.png` | Bull/Bear 방향 정확도 + 혼동행렬 |
| `04_simulation.png` | Long-only 전략 vs Buy & Hold 누적 수익률 |

---

## 디렉토리 구조

```
├── app.py                          Streamlit 대시보드
├── requirements.txt                패키지 설치
├── packages.txt                    시스템 패키지 (Streamlit Cloud용, libgomp1)
├── wsts_historical.xlsx            WSTS 원본 데이터
│
├── .github/workflows/
│   └── monthly-retrain.yml         분기별 자동 재학습 워크플로우
│
├── stage1/                         Stage 1 — 반도체 업황 YoY% 예측
│   ├── config.py
│   ├── pipeline.py
│   ├── s1_data.py                  Step 1: 데이터 수집 + 피처 엔지니어링
│   ├── s2_tune.py                  Step 2: Optuna RMSE 튜닝
│   ├── s3_select.py                Step 3: 다중공선성 제거 + RFE 피처 선택
│   ├── s4_optimize.py              Step 4: AsymLoss Bear 최적화
│   ├── s5_evaluate.py              Step 5: 최종 평가 + 시각화
│   └── outputs/
│       ├── data/                   CSV 데이터 파일
│       ├── models/                 학습된 모델 pkl ★ Stage 2 및 대시보드 입력
│       ├── figures/                시각화 png
│       └── metrics/                평가 지표 CSV
│
└── stage2/                         Stage 2 — SK하이닉스 주가 수익률 예측
    ├── config.py
    ├── pipeline.py
    ├── s1_dates.py                 Step 1: 분기 날짜 생성 + 타겟 수익률 산출
    ├── s2_data.py                  Step 2: A~E 피처 수집
    ├── s3_stage1_feat.py           Step 3: Stage 1 Expanding Window 예측
    ├── s4_features.py              Step 4: 피처 엔지니어링 + 피처 선택
    ├── s5_tune.py                  Step 5: XGBoost Optuna 튜닝
    ├── s6_evaluate.py              Step 6: 최종 평가 + 시각화
    └── outputs/
        ├── data/                   CSV 데이터 파일 ★ 대시보드 입력
        ├── models/                 학습된 모델 pkl ★ 대시보드 입력
        ├── figures/                시각화 png
        └── metrics/                평가 지표 CSV
```