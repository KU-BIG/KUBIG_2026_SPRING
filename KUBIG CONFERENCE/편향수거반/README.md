# Causal Debiasing for Recommender Systems
### Popularity Bias Diagnosis and Causal Debiasing on KuaiRec

---

## Motivation

추천시스템은 인기 있는 콘텐츠를 반복적으로 노출하는 경향이 있다. 이는 단순한 알고리즘 문제가 아니라, 관측 데이터 자체에 **노출 편향(exposure bias)** 이 내재돼 있는 구조적 문제다.

> **"인기 있어서 추천되는 건지, 추천돼서 인기 있는 건지 — 이건 인과관계의 문제다."**

이 프로젝트는 그 편향을 통계적으로 진단하고, 인과추론으로 교정하고, 실제로 작동하는 서비스로 만드는 과정이다.

---

## Project Identity

> 이 프로젝트는 **"추천 모델 성능 경쟁"** 이 아니라
> **"추천 로그의 편향을 통계적으로 분석하고 교정"** 하는 데 초점을 둔다.

| 키워드 | 설명 |
|--------|------|
| Popularity Bias | 인기 아이템이 더 노출되고, 더 클릭되는 악순환 |
| Exposure Skew | 노출 기회의 불평등 (Gini=0.672) |
| IPS Debiasing | 노출 확률로 학습 가중치 보정 |
| Causal Analysis | DML로 인과효과와 confounding 분리 |
| Evaluation Validity | 편향 없는 ground truth(small_matrix)로 공정 평가 |

---

## Dataset — KuaiRec

중국 쾌수(快手) 앱의 실제 추천 로그 데이터.

| 파일 | 설명 | 크기 |
|------|------|------|
| `big_matrix.csv` | 실제 서비스 노출 로그 (편향된 관측) | 12.5M rows |
| `small_matrix.csv` | 완전관측 실험 데이터 (unbiased GT) | 4.7M rows |
| `user_features.csv` | 유저 특성 56개 | 7,176명 |
| `item_categories.csv` | 아이템 카테고리 | 10,728개 |
| `item_daily_features.csv` | 아이템 일별 통계 | - |

**왜 KuaiRec인가:**
- `big_matrix`: 편향된 관측 → IPS 학습용
- `small_matrix`: 거의 완전관측 → 편향 없는 평가용
- 이 두 행렬의 분리 구조 덕분에 **공정한 debiasing 실험**이 가능하다

---

## Analysis Pipeline

### 1. Problem — 추천 로그는 biased하다

```
인간의 심리적 편향 (사회적 증거, 밴드왜건 효과)
        ↓
편향된 클릭·시청 데이터
        ↓
편향된 추천 모델 학습
        ↓
인기 아이템만 추천 → 악순환
```

> 데이터를 EDA로 말끔하게 만든다고 다가 아니다.
> **데이터 자체가 구조적으로 편향돼 있다.**

---

### 2. Diagnosis — 편향 정량화

| 지표 | 값 | 의미 |
|------|-----|------|
| Item Gini 계수 | **0.672** | 한국 소득 불평등(0.35)의 2배 수준 |
| 상위 10% 아이템 점유율 | **40.7%** | 소수 인기 아이템의 노출 독점 |
| Spearman ρ (인기도 vs watch_ratio) | **-0.299** | 인기 ↑ → 시청 만족도 ↓ |
| User Gini | 0.318 | 아이템 편향(0.672)이 2배 이상 심각 |

**핵심 발견:** 인기 있을수록 watch_ratio가 낮아지는 패턴 → 단순 상관관계인가, 인과관계인가? → Step 6에서 답한다.

---

### 3. Propensity Score 설계

ML 기반 P(T=1|X) 추정이 이상적이나, **KuaiRec big_matrix에 T=0(비노출) 케이스가 없어 불가능**하다. 대신 popularity 기반 근사를 사용:

$$p_i = \text{clip}\left(\frac{\text{pop}_i^{\alpha}}{\max_j \text{pop}_j^{\alpha}},\ 0.01,\ 1.0\right)$$

α를 [0.0, 0.5, 0.7, 1.0]으로 sweep하여 IPW weight 분산을 진단:

| α | weight 평균 | p95/p50 | 특징 |
|---|------------|---------|------|
| 0.0 | 1.000 | 1.000 | 보정 없음 (Baseline) |
| 0.5 | 3.433 | 1.537 | **균형점 (default)** |
| 0.7 | 5.740 | 1.825 | 강한 보정 |
| 1.0 | 11.977 | 2.362 | 매우 강한 보정 |

모든 α에서 p95/p50 < 10 → 학습 안정성 확보

---

### 4. Debiasing — IPS-debiased BPR

PyTorch로 BPR을 직접 구현하고, propensity를 손실함수의 가중치로 적용:

$$\mathcal{L}_{IPS} = -\frac{1}{n}\sum_{(u,i,j)} \frac{w_i}{\bar{w}} \cdot \ln\sigma(\hat{x}_{ui} - \hat{x}_{uj}) + \lambda\|\Theta\|^2$$

- $w_i = 1/p_i$: 롱테일 아이템에 더 큰 가중치
- `NORMALIZE_WEIGHTS=True`: batch 내 평균 weight=1 정규화 (SNIPS와 유사한 variance 안정화 효과)

---

### 5. Evaluation Results

| α | Recall@20 | NDCG@20 | Coverage@20 | Gini@20 |
|---|-----------|---------|------------|---------|
| 0.0 (Baseline) | 0.0156 | 0.7568 | 0.0060 | 0.9940 |
| 0.5 | 0.0161 | 0.7967 | 0.0081 | 0.9939 |
| 0.7 | 0.0165 | 0.8209 | 0.0105 | 0.9934 |
| **1.0** | **0.0168** | **0.8237** | **0.0114** | **0.9936** |

**α=1.0 기준 Baseline 대비:**
- Recall@20: **+7.8%**
- NDCG@20: **+8.9%**
- Coverage@20: **+90%**

> 일반적으로 편향 보정은 "정확도↓ + 다양성↑" trade-off가 발생한다.
> 그런데 여기선 **정확도와 다양성이 동시에 향상**됐다. 이유는 Step 6에서 밝힌다.

---

### 6. Causal Analysis — Double Machine Learning

**연구 질문:** X(유저·아이템 특성)를 통제했을 때, 인기도 T가 watch_ratio Y에 미치는 순수 효과 θ는?

**설계:**
- Y = watch_ratio (cap=2.0), small_matrix
- T = z-score(log(1+노출횟수)), 연속형
- X = user features + video_duration
- 5-fold cross-fitting (Chernozhukov et al., 2018)
- Nuisance 모델: XGBoost

**결과:**

| 방법 | θ | p-value | 해석 |
|------|---|---------|------|
| Naive (통제 없음) | -0.0649 | < 0.001 | 강한 음의 상관 |
| **DML (X 통제)** | **+0.0021** | **0.349** | **0과 구분 안 됨** |

**Nuisance R² 진단:**
- T ~ X R² = **0.067** (낮음 → 인기도는 특성과 거의 무관)
- Y ~ X R² = **0.431** (높음 → 시청률은 특성으로 잘 설명됨)

**핵심 발견:**
> 음의 상관관계(-0.0649)는 **100% confounding**이었다.
> 인기도 자체엔 인과효과가 없다 (θ≈0, p=0.349).

**이게 Step 5 결과를 설명한다:**
> Baseline은 "인과효과 없는 인기도 신호"를 따라 underfitting.
> IPS가 그 신호를 약화시키니 더 의미있는 패턴을 학습 → 정확도까지 향상.

---

### Causal DAG

```
        X (user · item features)
       ↙ R²=0.067        ↘ R²=0.431
      T (popularity)  ----→  Y (watch_ratio)
                    θ≈0
              (p=0.349, 유의하지 않음)

실선: confounding path
점선: DML 추정 대상 (deconfounded partial effect)
```

---

## Limitations

1. **Propensity 이론적 한계**
   - popularity 기반 근사 사용 (T=0 데이터 없어 ML 기반 불가)
   - 유저·아이템 특성을 고려한 P(T=1|X) 추정 불가

2. **단일 seed 실험**
   - 결과의 분산을 확인하지 못함
   - 여러 seed 평균이 이상적

3. **평가 셋업의 구조적 한계**
   - small_items(3,327개) 내에서만 ranking
   - Baseline이 평가 범위 밖 아이템에 집중했을 가능성

4. **DML 표현 주의**
   - popularity(T)는 내생변수(endogenous variable)
   - "진짜 causal effect"가 아닌 "deconfounded partial effect estimate"

---

## Future Work

| 항목 | 설명 |
|------|------|
| SNIPS / DR Estimator | variance 안정화, doubly robust 추정 |
| ML 기반 Propensity | T=0 레이블 확보 시 P(T=1\|X) 직접 추정 |
| Time-based split | 시간 축 기반 train/test 분리 |
| ESS 기반 α 선택 | Effective Sample Size로 α 자동 탐색 |
| Multi-label 카테고리 | feat 컬럼의 다중 카테고리 활용 |

---

## Tech Stack

```
Data:     pandas, numpy
Model:    PyTorch (BPR 직접 구현)
Causal:   statsmodels (OLS/HC1), XGBoost (nuisance)
Eval:     scikit-learn
Viz:      matplotlib
Service:  Streamlit
```

---

## Quick Start

```bash
# 설치
pip install -r requirements.txt

# Streamlit 실행
streamlit run app.py
```

**필요 파일 (data/ 폴더):**
```
data/
├── big_matrix.csv
├── small_matrix.csv
├── item_categories.csv
├── item_daily_features.csv
├── model_data.pkl
├── dml_results.pkl
└── step3_results.csv
```

---

## Streamlit Dashboard

| 탭 | 내용 |
|----|------|
| IPS 편향 보정 시연 | 유저 선택 + λ 슬라이더 → Baseline vs IPS 추천 비교 |
| Bias Dashboard | Lorenz curve, Gini, α sweep 시각화 |
| DML Analysis | Naive vs DML θ 비교, confounding 시각화 |

## Live Demo
[Streamlit App](https://h8c63pzauiyhamesmzdqdu.streamlit.app/)

---

## References

- Chernozhukov et al. (2018). *Double/Debiased Machine Learning.* Econometrics Journal.
- Schnabel et al. (2016). *Recommendations as Treatments: Debiasing Learning and Evaluation.*
- Gao et al. (2022). *KuaiRec: A Fully-observed Dataset for Recommender Systems.*
- Rendle et al. (2009). *BPR: Bayesian Personalized Ranking.*
