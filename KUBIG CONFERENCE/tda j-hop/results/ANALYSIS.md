# 종합 분석 — 위상 특징 × 이종 그래프 노드분류 (전 조건 · paired 검정)

> 조건: (a)=HAN, (b1)=HAN+random noise, (b2)=HAN+class-mix, (c)=HAN+real 위상,
> (d)=RGCN, (e1)=RGCN+noise, (e2)=RGCN+class-mix, (f)=RGCN+real 위상.
> random seed 10개, mean±std. **(f)는 5/7** (aifb·yelp 미완; mag 는 요약치만).
> 수치: `SUMMARY.md` · per-seed paired 검정: **`paired_stats.md`** (Wilcoxon signed-rank, 95% CI, win-rate).

## 0. 최종 결과표 (test Macro-F1)

| 데이터셋 | (a) HAN | (b1) +noise | (b2) +mix | (c) +real | (d) RGCN | (e1) +noise | (e2) +mix | (f) +real |
|---|---|---|---|---|---|---|---|---|
| acm | 0.895±0.006 | 0.887±0.008 | 0.899±0.006 | 0.894±0.008 | 0.925±0.006 | 0.919±0.013 | 0.923±0.012 | 0.903±0.006 |
| dblp | 0.786±0.013 | 0.776±0.010 | 0.878±0.018 | 0.862±0.013 | 0.934±0.004 | 0.930±0.008 | **0.939±0.005** | 0.859±0.017 |
| imdb | 0.438±0.009 | 0.441±0.004 | 0.447±0.021 | 0.450±0.019 | **0.636±0.004** | 0.579±0.008 | 0.634±0.004 | 0.622±0.004 |
| freebase | 0.146±0.055 | 0.160±0.059 | 0.162±0.027 | 0.144±0.051 | **0.209±0.108** | 0.158±0.061 | 0.207±0.056 | 0.166±0.065 |
| mag | 0.017±0.012 | 0.026±0.011 | 0.019±0.009 | 0.023±0.007 | **0.104±0.053** | 0.090±0.034 | 0.088±0.048 | 0.066±0.036 |
| aifb | 0.451±0.040 | 0.478±0.100 | 0.538±0.106 | 0.575±0.145 | **0.752±0.018** | 0.673±0.071 | 0.720±0.134 | – |
| yelp | 0.110±0.028 | 0.094±0.026 | 0.079±0.024 | 0.091±0.024 | 0.055±0.006 | 0.056±0.006 | 0.067±0.019 | – |

(accuracy 보조표는 `SUMMARY.md` §보조 지표. mag: acc 0.14→0.28, yelp: 0.62→0.87 로 두 데이터셋의
낮은 macro-F1 은 metric 특성 — 349클래스 subsample 평균 / 희귀 멀티라벨.)

---

## 1. Q1 — 위상 특징은 성능 이점을 주는가?

### 1.1 Macro-F1: paired per-seed 검정 결과 (`paired_stats.md`)

**HAN 위 (c)−(a):** 7개 중 **유의 3개** — dblp **+0.076** (win 100%, p=0.002), aifb **+0.124**
(win 90%, p=0.020), imdb +0.012 (p=0.049, 경계). acm/freebase/mag/yelp 는 무효과.
단 8대조×7데이터셋의 다중비교라 p≈0.05 경계(imdb)는 과신 금지.

**RGCN 위 (f)−(d):** 측정 4/4 **전부 유의하게 음수** — acm −0.022, dblp −0.075, imdb −0.014,
freebase −0.048 (win 0~11%, p=0.002~0.039). mag 도 요약치 기준 −0.038. **강한 백본에는
위상 주입이 통계적으로 확실한 손해.**

**대조군이 이득의 '정체'를 밝힘:**
- (b1)−(a) noise: 이득 없음 (acm 은 오히려 유의한 −0.008) → 차원 추가 효과 배제.
- **(b2)−(c) mix−real (HAN): dblp +0.016 (p=0.027) — 정렬을 깨면 오히려 유의하게 좋아짐.**
- **(e2)−(f) mix−real (RGCN): acm +0.021, dblp +0.079, imdb +0.012 모두 p=0.002·win 100%**
  (freebase +0.023, p=0.098). 같은 위상 값에서 per-node 정렬만 깨면 일관되게 상승.

→ per-node 정렬은 **기여 0 또는 음(音)의 기여**. 위상 이득이 있는 곳에서도 그 가치는
class-level 분포다 ([Lee et al., ICML 2024](https://arxiv.org/abs/2402.04621) 의 A–X 의존성
노이즈 제거 패턴과 정합).

### 1.2 Accuracy 관점
방향 동일: dblp acc +0.073(c vs a), aifb +0.045. mag/yelp 는 조건 간 차이가 분산 내.
imdb 는 위상 조건에서 accuracy std 0.002→0.110 — 일부 seed 학습 붕괴(불안정성 신호).

### 1.3 Robustness 관점 — 이점 없음
- **seed 분산(std)**: 개선 1~2 / 악화 2~3 / 나머지 ≈. aifb 는 (c)에서 std **3.6×** 악화(0.040→0.145).
- **worst-seed(최악 seed 성능)**: dblp 는 개선(0.766→0.838)이나 aifb 는 악화(0.438→0.387).
  freebase 는 (a)(c)(d) 모두 **macro-F1=0.000 인 붕괴 seed** 존재 — 유일하게 (f)만 붕괴 없음(min 0.153),
  그러나 평균은 (f)<(d)라 "위상=안정화" 주장으로는 부족.
- 초기 3-seed 의 "위상=분산 감소" 힌트는 10-seed 에서 재현 안 됨.
- (여기서의 robustness = seed 분산. 입력 교란/적대 내성은 미실험.)

### 1.4 판정
**무조건적 이점: 없음. 조건부 이점: "HAN × (dblp, aifb)" 두 곳뿐이고, 그 실체도 per-node
위상 정보가 아니다** (dblp=관계 커버리지 누출 §2.2-A, aifb=featureless 구조 인코딩 §2.2-B).
강한 백본(RGCN)에서는 유의한 손해. Robustness 이점도 없음.

---

## 2. Q2 — 데이터셋별 차이와 그 원인

### 2.1 특성 ↔ 효과 대응표

| 데이터셋 | feature | 타겟 N | 클래스 | 라벨 | 특이사항 | HAN 위상효과(paired) | 원인 분류 |
|---|---|---|---|---|---|---|---|
| acm | **강(1902d)** | 3,025 | 3 | 단일 | 인용, homophily 2.0× | ≈0 (p=0.92) | C. feature 포화 |
| dblp | 중(334d) | 4,057 | 4 | 단일 | venue 가 클래스 결정 | **+0.076\*** | **A. 커버리지 누출** |
| imdb | 강(3489d) | 4,932 | 5 | **멀티** | GTN NaN 진단, acc 불안정 | +0.012 (경계) | C. feature 포화 |
| freebase | 없음 | 40k→6k sub | 7 | 단일 | **붕괴 seed(F1=0) 존재**, mixed 9.9% | ≈0 | E. 분산 지배 |
| mag | 중(128d) | 12k→3k sub | **349** | 단일 | macro-F1 붕괴 | ≈0 | D. metric 붕괴 |
| aifb | **없음** | 2,270 | 4 | 단일 | 소형·깨끗한 RDF, mixed 7.7% | **+0.124\*** | **B. 구조 인코딩** |
| yelp | 없음 | 5,484 | 16 | **멀티** | 3.9M 엣지 초밀집 | −0.020 | D+E |

### 2.2 메커니즘 분해 (b2_manual 통제 — HAN 메타패스 2개 *위에서만* 계산한 위상)

| | (a) | b2_manual | (c) GTN 위상(전 관계) | 결론 |
|---|---|---|---|---|
| dblp | 0.786 | **0.790 (+0.004)** | 0.862 (+0.076) | 이득 = 위상 아님, **누락 관계(venue) 주입** |
| aifb | 0.451 | **0.560 (+0.109)** | 0.575 (+0.124) | 이득 = **구조 인코딩** (같은 관계 위상만으로 대부분 재현) |
| acm | 0.895 | 0.896 | 0.894 | 아무것도 안 통함 |

**A. 관계 커버리지 누출 (dblp).** 클래스(연구분야)를 결정하는 venue 관계(APVPA)가 HAN 메타패스
(APA·APTPA)에 빠져 있고 GTN 채널(전 관계)에는 있음. 같은 위상을 HAN 의 2개 메타패스 위에서
계산하면 이득이 +0.004 로 소멸 → "+0.076 위상 이득"의 실체는 **EPD 가 아니라 GTN 채널을 경유한
누락 관계 정보**. mix>real(위상 내용이 아니라 class-상관 분포만 중요), RGCN(venue 직접 사용)
0.934 와 모두 정합.

**B. featureless 구조 인코딩 (aifb).** one-hot feature 뿐이라 *어떤* 노드 구조 기술자도 정보가 됨.
같은 관계 위 위상만으로 +0.109. paired 로는 유의(p=0.020, win 90%)하나 **std 0.145 로 매우
불안정**하고, (c)−(b1)은 비유의(p=0.275) — "noise 이상의 가치"는 통계적으로 확립 못 함.

**C. feature 포화 (acm·imdb).** 1902~3489 차원 bag-of-words 가 클래스를 사실상 결정. 메타패스
homophily 2.0×(실측)로 구조 신호는 있으나 feature 와 중복 → 위상·noise·mix 전부 ≈0.

**D. metric 붕괴 (mag·yelp).** 349클래스 subsample 평균 / 희귀 멀티라벨 → macro-F1 이 신호를
못 담음(accuracy 는 정상 학습 증명). 조건 간 Δ 만 유효.

**E. 분산 지배 (freebase, yelp).** freebase 는 (a)(c)(d)에 **macro-F1=0.000 붕괴 seed** 가 있고
std ±0.05~0.11 — 어떤 조건 효과도 분산에 묻힘. featureless 라도 aifb 처럼 작동하지 않는 이유:
subsample 후 구조 훼손 + 라벨 희소. → **"featureless 면 위상이 돕는다"가 아니라
"featureless + 깨끗한 구조 + 조밀한 라벨(aifb)"** 이 정확한 조건.

### 2.3 위상 자체의 관점
- 시각화 실측에서 **대부분 노드의 ego EPD 는 위상적으로 빈약**(near-clique ego → 유한 위상점 0,
  degenerate 다수) — per-node 위상이 노드를 구분할 정보량 자체가 제한적.
- 메타패스 채널은 class-homophilous(acm 2.0×)하므로 **채널 구조에 class 신호는 있다** — 그러나
  그 신호는 GNN 메시지패싱이 이미 소비하는 것과 중복이고, EPD 요약이 *추가로* 주는 부분이
  aggregate 지표에서 검출되지 않는 것.

---

## 3. Q3 — HAN vs RGCN, 그리고 메타패스는 NC 에 적합한가

### 3.1 백본 비교 (paired (d)−(a): 7/7 전부 유의)
acm +0.029, dblp **+0.148**, imdb **+0.198**, freebase +0.063, mag +0.087, aifb **+0.301**
(모두 win 90~100%, p≤0.008). yelp 만 macro-F1 −0.055\* 인데 accuracy 는 0.624→0.871 —
희귀 라벨을 버리고 다수를 맞히는 trade-off (지표 착시). **accuracy 기준 RGCN ≥ HAN 7/7.**

| 요인 | 내용 |
|---|---|
| 입력 커버리지 | RGCN=base relation 전부(3–6), HAN=메타패스 2개. dblp 격차 +0.148 의 대부분이 venue 누락(§2.2-A) — **공정성 caveat**: 순수 아키텍처 비교 아님 |
| 표현력 | 관계별 전용 W_r vs 공유변환+스칼라 β (메타패스 2개면 attention 이 고를 것도 없음) |
| 홈그라운드 | RDF entity 분류는 RGCN 원논문 벤치마크 (aifb +0.301) |
| noise 내성 | 역설: **HAN 은 noise 둔감, RGCN 은 유의하게 민감**((e1)−(d): imdb −0.057\*, freebase −0.051\*, aifb −0.079\*) — 그럼에도 RGCN 이 전부 이김 |
| 보조 증거 | dblp 에서 GTN-only(0.920) > HAN+위상(0.862) — 병목은 HAN+메타패스 부분집합 자체 |

### 3.2 메타패스는 NC 에 도움이 되었나
**판정: 제한적으로만. 관계-완전 메시지패싱에 지배당함(dominated).**
- 도움: 메타패스 그래프는 class-homophilous(2.0×) — HAN 이 작동하는 이유.
- 한계: ① **선택(coverage)이 아킬레스건** — dblp 메타패스 2개가 venue 누락 → −0.148.
  ② 관계-완전 1-hop(RGCN)이 전 데이터셋 우세 → 경로 사전합성이 불필요.
  ③ GTN 으로 경로를 *학습*해도 수동 선택 대비 이득 없음(acm: 0.894 vs b2_manual 0.896) —
  GTN 의 실질 기여는 경로 발견이 아니라 **전 관계 접근**.
  ④ 사전합성+kNN 희소화 과정에서 엣지 중복도·가중치 소실.
- **메타패스의 실제 가치 = 해석성** (어떤 의미 경로가 노드를 묶는지; 시각화로 확인 가능).

---

## 4. 추가 발견·진단

1. **noise 민감성의 백본 비대칭** — HAN 은 무의미한 75차원에 둔감(projection+attention 이 걸러냄),
   RGCN 은 유의하게 취약(3/7 데이터셋 유의 하락). 아키텍처의 입력 선택성 차이. 실무 함의:
   RGCN 계열에 feature 를 얹을 땐 사전 선별 필요.
2. **class-mix 의 mixed-ratio caveat** — 구현이 train/val/test split 구성원(=라벨 노드)만 셔플:
   acm/dblp 100%, imdb 93%, mag 98%, yelp 94% 는 충분하나 **freebase 9.9%·aifb 7.7%만 혼합**
   (라벨 노드가 소수). 이 두 데이터셋에서 mix≈real 은 대조군이 약해서일 수 있음 — 단 평가 노드
   자신의 위상 정렬은 파괴되므로 "own-feature 정렬" 검정으로는 유효.
3. **freebase 붕괴 seed** — (a)(c)(d)에 macro-F1=0.000 seed 존재(멀티클래스 전붕괴). freebase 의
   모든 조건 비교는 신뢰도 낮음. (f)만 붕괴 없음(min 0.153)은 흥미로우나 평균이 낮아 해석 보류.
4. **imdb·freebase 의 GTN attention NaN — 근본 원인 규명**: NaN 은 정확히 **모든 base relation
   에서 완전 고립된 타겟 노드가 있는 두 데이터셋**(imdb 61개=1.2%, freebase 97개=1.6%; 나머지는
   0개)에서만 발생. 로그상 ep1 loss 정상 → ep10 내 NaN. 메커니즘: 고립 노드는 identity 관계의
   softmax 가중치로만 채널 degree 를 갖는데, 학습 중 그 가중치가 줄면 deg→0 → `_gtn_norm` 의
   `1/deg` 폭발(+`torch.where` backward 0×inf=NaN) → GTConv 가중치 전체 NaN 고착.
   (멀티라벨 미지원이 원인 아님 — freebase 는 단일라벨인데 NaN, yelp 는 멀티라벨인데 정상.)
   **파급**: 이 두 데이터셋의 (c)/(f) GTN 채널은 사실상 초기(미학습) 혼합 — 채널 품질 caveat.
   수정안: `deg.clamp(min=ε)` / GTN grad clipping / 고립 노드 제거.
   **후속 검증(로그) — 실제로는 "0벡터 위상"**: NaN run 의 stage-2 로그에서 HKS 고유값이
   전부 λ=1(=엣지 0개 그래프), PDGNN 학습 라인 없음, stage2 3~5초(정상 ~1,000초). 즉
   H(NaN)→kNN 이웃 0→빈 채널→HKS 상수→ego 빈 그래프→**위상 특징 전량 0**. 따라서
   **imdb·freebase 의 위상 조건 결과는 위상에 대한 증거가 아니며**(silent failure),
   imdb (c)−(a) 경계 유의(+0.012, p=0.049)는 허위 신호로 기각, freebase (f) 무붕괴 관찰도
   무효. 핵심 결론(acm·dblp·aifb 기반 mix>real·f<d·커버리지 분해)은 GTN 정상이라 불변.
   미해결: 협업자 imdb e2 vs f 유의차(0.634 vs 0.622)는 둘 다 0벡터라면 불가능 — 협업
   환경 로그 확인 필요.
5. **e2>d (dblp +0.005)**: 셔플된 위상조차 RGCN 을 살짝 올리는 유일한 사례 — class-level 위상
   분포가 순수 feature 로서 미미하게 기여할 수 있음을 시사(유의성 미검정).

---

## 5. 종합 결론

1. **GTN-PDGNN per-node 위상 특징은 이종 NC 에서 instance-level 기여가 없다** — real≈mix(HAN),
   **mix>real 유의(RGCN 3/4, p=0.002)**, 강한 백본에선 주입 자체가 유의한 손해(f<d 4/4\*).
2. 위상이 aggregate 를 올리는 두 사례(dblp·aifb)는 각각 **관계 커버리지 누출**과 **featureless
   구조 인코딩**으로 분해되며, 둘 다 "EPD 위상이라서"가 아니다.
3. **백본 선택이 위상 주입보다 훨씬 크다** — RGCN ≥ HAN (accuracy 7/7, paired 전부 유의).
   메타패스 기반 모델의 병목은 메타패스 *선택*이며, 관계를 전부 주는 모델이 우선.
4. 본 프로젝트 선행 관찰("통제하면 위상 신호 없음")이 **2 백본 × 4 조건 × 10 seed × paired 검정**
   으로 승격·재확인됨.

## 6. 한계

- aggregate metric(macro-F1/acc)은 within-class 정보에 둔감 — real≈mix 단독으론 per-node 무의미
  단정 불가(경계). 반면 **mix>real 은 방향성 있는 유의 결과**로 더 강함. per-node metric
  (margin/NLL)·η² 진단 미수행.
- (f) 5/7 (aifb·yelp 미완, mag 은 요약치만·per-seed 없음). 다중비교 보정 미적용(경계 p 주의).
- 백본 비교는 입력 비대칭 하의 "표준 사용법" 비교(§2.2-A 가 dblp 에서 그 크기를 정량화).
- class-mix 의 freebase/aifb 혼합률 저하(§4-2), freebase 붕괴 seed(§4-3), imdb GTN NaN(§4-4).

## 7. 향후 과제 — 위상 × Link Prediction

근거: 본 연구의 실패 지점은 per-node *정렬*인데 LP 는 **노드쌍 지역 구조**가 직접 신호이고,
EPD 의 루프(H1)는 링크 형성(triadic closure)과 기계적으로 연결된다. TLC-GNN(원류)·PDGNN 논문
모두 LP 에서 우호적 결과 보고. **단** 본 프로젝트의 선행 hetero-LP 통제 실험은 null 이었으므로,
재도전 시 이번 교훈 필수 반영: pair-level permutation 대조군(같은 공통이웃/거리 bucket 내 셔플),
관계 커버리지 통제(§2.2-A 누출 방지), featureless 세팅 우선, hard-pair subset metric.

## 8. 참고

- Yun et al., NeurIPS 2019 (GTN) · Yan et al., NeurIPS 2022 (PDGNN) · Wang et al., WWW 2019 (HAN)
- Schlichtkrull et al., ESWC 2018 (RGCN) · Yan et al., ICML 2021 (TLC-GNN)
- Lee et al., ICML 2024, arXiv:2402.04621 (within-class shuffle) · Strobl et al., 2008 (conditional permutation)
- 부속 파일: `paired_stats.md`(paired 검정 전체) · `CLASS_WISE_MIXING.md`(mix 상세) ·
  `f_rgcn_per_seed.md`((f) per-seed) · `SUMMARY.md`(전 표)
