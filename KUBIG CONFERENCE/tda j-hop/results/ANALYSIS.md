# 최종 분석 — 위상 특징 · 메타패스 · 이종 그래프 노드분류

> **설계**: 2 백본(HAN/RGCN) × 4 위상조건(없음 / random-noise / class-mix / real GTN-PDGNN) × 7 데이터셋 × 10 random seed.
> (a)=HAN, (b1)=+noise, (b2)=+class-mix, (c)=+real | (d)=RGCN, (e1)=+noise, (e2)=+class-mix, (f)=+real.
> imdb·freebase 의 위상 조건은 GTN NaN 버그(§5-F7) 수정 후 재실행한 유효값. (f)의 aifb·yelp 만 미완.
> `*` = paired Wilcoxon signed-rank p<0.05 (seed 짝, n=10; `paired_stats.md`). 표: `SUMMARY.md`.

## 0. 결과 요약표 (test Macro-F1, mean±std)

| 데이터셋 | (a) HAN | (b1) +noise | (b2) +mix | (c) +real | (d) RGCN | (e1) +noise | (e2) +mix | (f) +real |
|---|---|---|---|---|---|---|---|---|
| acm | 0.895±0.006 | 0.887±0.008 | 0.899±0.006 | 0.894±0.008 | 0.925±0.006 | 0.919±0.013 | 0.923±0.012 | 0.903±0.006 |
| dblp | 0.786±0.013 | 0.776±0.010 | 0.878±0.018 | 0.862±0.013 | 0.934±0.004 | 0.930±0.008 | **0.939±0.005** | 0.859±0.017 |
| imdb | 0.438±0.009 | 0.441±0.004 | 0.402±0.013 | 0.402±0.015 | **0.636±0.004** | 0.579±0.008 | 0.575±0.033 | 0.566±0.035 |
| freebase | 0.146±0.055 | 0.160±0.059 | 0.141±0.060 | 0.138±0.049 | **0.209±0.108** | 0.158±0.061 | 0.170±0.077 | 0.169±0.059 |
| mag | 0.017±0.012 | 0.026±0.011 | 0.019±0.009 | 0.023±0.007 | **0.104±0.053** | 0.090±0.034 | 0.088±0.048 | 0.066±0.036 |
| aifb | 0.451±0.040 | 0.478±0.100 | 0.538±0.106 | 0.575±0.145 | **0.752±0.018** | 0.673±0.071 | 0.720±0.134 | – |
| yelp | 0.110±0.028 | 0.094±0.026 | 0.079±0.024 | 0.091±0.024 | 0.055±0.006 | 0.056±0.006 | 0.067±0.019 | – |

accuracy 보조표는 `SUMMARY.md` (MAG·yelp 의 낮은 macro-F1 은 metric 특성 — acc 로는 각각 0.28 / 0.87 로 정상 학습).

---

## 1. Q1 — 위상 특징은 노드분류에 어떤 영향을 주었나? 왜 성능·robustness 가 나빠졌나?

### 1.1 영향의 전모 (paired 검정 기준)

| 방향 | 근거 |
|---|---|
| **이득 (2곳, HAN 만)** | dblp (c)−(a)=+0.076\*, aifb +0.124\* |
| **무효과 (다수)** | acm·mag·yelp (HAN), freebase(≈0) |
| **손해 (일관, RGCN 전부 + HAN imdb)** | (f)−(d): acm −0.022\*, dblp −0.075\*, imdb −0.070\*, freebase −0.040(p=.066), mag −0.038 · HAN imdb (c)−(a)=−0.036\* |
| **per-node 정렬 무기여** | mix≥real: (e2)−(f) acm +0.021\*, dblp +0.079\*; (b2)−(c) dblp +0.016\*; 그 외 Δ≈0. **real 이 mix 를 유의하게 이긴 곳 = 0/7** |

그리고 이득 2곳조차 분해하면 위상 고유의 공이 아니다(§3.2): dblp = 관계 커버리지 누출, aifb = featureless 구조 인코딩.

### 1.2 성능이 나빠진 이유 — 증거로 지지되는 4개 메커니즘

**(i) 무게이트(gating 없는) 백본의 노이즈 채널 효과.** 순수 random noise 만으로 RGCN 이 유의하게 하락
((e1)−(d): imdb −0.057\*, freebase −0.051\*, aifb −0.079\*). RGCNConv 는 concat 입력 전체를 관계별
선형변환에 통과시켜 무의미한 75차원이 모든 메시지에 분산을 주입한다. 반면 HAN 은 projection+
attention 이 등방 noise 를 걸러 (b1)≈(a). → 하락분의 일부는 "위상"이 아니라 **차원 추가 자체**.

**(ii) 구조-상관 잡음은 등방 잡음보다 해롭다.** imdb 에서 real 위상이 **noise 보다도 유의하게 나쁨**
((c)−(b1)=−0.039\*, freebase −0.022\*). 등방 noise 는 이웃 집계에서 평균화되어 소거되지만, 위상
특징은 그래프 구조와 상관(같은 이웃권 노드는 비슷한 ego EPD)이라 **집계 후에도 살아남아** 잘못된
결정 규칙을 만든다. 이는 구조–특징(A–X) 의존성이 그래프 컨볼루션의 실효 잡음을 늘리고, within-class
shuffle 로 그 의존성을 줄이면 성능이 오른다는 Lee et al. (ICML 2024) 의 예측과 정확히 일치 —
실제로 우리 (e2)>(f)\*, (b2)>(c)\*(dblp) 가 그 패턴이다.

**(iii) 신호 자체의 빈곤.** 시각화 실측에서 대부분 노드의 ego EPD 는 위상적으로 퇴화(near-clique →
유한 위상점 0). 남는 정보도 class-level 분포뿐임이 mix 대조로 확인됨(§1.1). 즉 유효 신호가 이미
백본이 소비하는 구조 정보와 중복 → 순기여 없이 (i)(ii)의 비용만 남는다.

**(iv) 표현(featurization) 병목.** persistence image 의 pers_range(0,6)가 실측 persistence(~0–0.3)
대비 ~20배 넓어 25차원 중 유효 해상도가 크게 저하(실측 진단). 채널 자체는 문제 아님 — 픽스 후
imdb 에서 **GTN-only F1=0.525 > HAN 0.438** 인데도 그 채널의 EPD 를 HAN 에 주입하면 0.402 로
하락: **손실은 채널→EPD→PI→concat 경로에서 발생**한다.

### 1.3 robustness(=seed 분산)가 나빠진 이유

관측: aifb (c) std 0.145 vs (a) 0.040 (**3.6×**), worst-seed 0.387<0.438; imdb (f) std 0.035 vs (d)
0.004 (**~9×**), worst-seed 0.475<0.631; dblp (f) 0.017 vs (d) 0.004. 초기 3-seed 의 "위상=분산 감소"
힌트는 10-seed 에서 기각.

메커니즘: 위상 특징은 **그 자체가 seed 의존 확률변수**다 — GTN 초기화/학습(채널), PDGNN 샘플링/
학습, fusion 이 모두 seeded 라 *입력 표현*에 추가 확률성이 곱해진다(stage 3단 파이프라인의 분산
합성). 모델이 그 특징에 의존할수록(featureless aifb) 또는 백본이 입력에 민감할수록(RGCN) 이 분산이
출력으로 전파된다. 특징을 무시하는 곳(feature 포화 acm)은 분산 불변 — 의존도와 분산 악화가 정확히
동행한다는 점이 이 설명의 근거.

---

## 2. Q2 — 메타패스(HAN) 는 노드분류에 이득을 주었나?

**아니오 — 관계-완전(relation-complete) 메시지패싱(RGCN)에 전면 지배(dominated)된다.**

- (d)−(a) paired: **7/7 유의** (dblp +0.148\*, imdb +0.198\*, aifb +0.301\*, …; yelp 만 macro-F1 −0.055\*
  이나 accuracy 0.624→0.871 로 지표 착시). **accuracy 기준 RGCN ≥ HAN 7/7.**
- 열세의 최대 원인 = **메타패스 '선택'**: dblp 클래스(연구분야)를 결정하는 venue 관계(APVPA)가 HAN
  메타패스(APA·APTPA)에 누락 — RGCN(전 관계+관계별 W_r)이 이를 직접 사용. 같은 위상 파이프라인을
  HAN 의 2개 메타패스 위에서만 돌리면 이득이 +0.004 로 소멸(§3.2)하는 것이 정량 증거.
- GTN 으로 메타패스를 **학습**해도 구제 불가: acm 에서 GTN(0.894) ≈ 수동(0.896); GTN 의 실질 기여는
  경로 발견이 아니라 **전 관계 접근**이다. imdb 에선 GTN 단독(0.525)이 HAN(0.438)을 이김 — 병목은
  경로 사전합성+부분집합 구조의 HAN 쪽.
- **남는 가치 = 해석성**: 메타패스 채널은 class-homophilous(acm 실측 same-class edge 0.67 vs 무작위
  0.33, **2.0×**) — "어떤 의미 경로가 노드를 묶는가"의 설명 도구로는 유효.
- caveat: 입력 비대칭(RGCN 관계 3–6 vs HAN 메타패스 2) 하의 "표준 사용법" 비교임. 단 그 비대칭
  자체가 메타패스 방법론의 본질적 약점(선택 위험)이다.

---

## 3. Q3 — 데이터 특성에 따라 결과가 달라졌나? 왜?

### 3.1 특성 → 결과 대응

| 데이터셋 | 결정적 특성 | 위상 효과(HAN, paired) | 원인 유형 |
|---|---|---|---|
| acm | feature 강(1902d), homophily 2.0× | ≈0 | **C 포화** |
| imdb | feature 강(3489d), 멀티라벨 | **−0.036\*** (noise 보다도 ↓) | **C 포화 + (ii) 구조상관잡음** |
| dblp | feature 중(334d), venue 가 클래스 결정 | +0.076\* | **A 커버리지 누출** |
| aifb | **featureless**, 소형·깨끗한 RDF | +0.124\* (불안정 ±0.145) | **B 구조 인코딩** |
| freebase | featureless, subsample·희소 라벨, 붕괴 seed | ≈0 | **E 분산 지배** |
| mag | 349클래스 subsample | ≈0 | **D metric 붕괴** |
| yelp | featureless·초밀집(3.9M)·희귀 멀티라벨 | ≈0/− | **D+E** |

### 3.2 메커니즘 검증 (b2_manual = HAN 메타패스 위상만 사용, 아카이브 10-seed)

| | (a) | +위상(HAN 메타패스만) | +위상(GTN 전 관계) | 해석 |
|---|---|---|---|---|
| dblp | 0.786 | 0.790 (+0.004) | 0.862 (+0.076) | 이득 = **누락 관계(venue) 주입**, EPD 아님 |
| aifb | 0.451 | 0.560 (+0.109) | 0.575 (+0.124) | 이득 = **featureless 보완용 구조 기술자** |
| acm | 0.895 | 0.896 | 0.894 | 아무것도 안 통함 |

**일반화**: 위상(및 어떤 구조 기술자든)이 도우려면 ① node feature 가 약/부재하고, ② 구조가
깨끗하며(subsample 훼손·고립 없음), ③ 라벨이 조밀하고, ④ 지표가 신호를 담을 수 있어야 한다 —
7개 중 이를 다 만족한 것은 aifb 뿐. "featureless 면 위상이 돕는다"는 부정확하고, **"위상은
'차선의 feature 대체재'이며 진짜 feature 나 전체 관계 접근이 있으면 즉시 잉여·유해로 전락한다"**
가 정확한 요약. 메타패스 효과의 데이터 의존성도 동일 축: 클래스-결정 관계가 메타패스에 포함되면
작동(acm), 누락되면 실패(dblp).

---

## 4. Q4 — 추후 과제: 위상 특징·메타패스는 어떤 task 에 효과가 있을까?

**(1) Link prediction (가장 유력).** 본 연구의 실패 지점은 per-node "정렬"인데, LP 는 **노드쌍의
지역 구조**가 직접 신호이고 EPD 의 H1(루프)은 triadic closure 와 기계적으로 연결된다. 실제로 이
파이프라인의 원류인 TLC-GNN (Yan et al., ICML 2021)은 PH 기반 **LP** 방법이고, PDGNN 논문(Yan et
al., NeurIPS 2022)도 LP 벤치마크에서 true-EPD 동급 성능을 보고했다. **단** 본 프로젝트의 선행
hetero-LP 통제 실험은 null 이었으므로, 재도전 시 필수 설계: pair-level permutation 대조군(공통이웃/
거리 bucket 내 셔플 — 본 연구 class-mix 의 LP 판), 관계 커버리지 통제(§3.2-A 누출 방지),
featureless 세팅 우선, hard-pair subset 지표.

**(2) Graph-level 분류.** persistence 는 본질적으로 전역 요약이라 그래프 단위 비교(분자·단백질 등)
에서 판별력이 산다 — PH 기반 그래프 분류는 확립된 영역(Hofer et al., NeurIPS 2017; PersLay,
Carrière et al., AISTATS 2020; TOGL, Horn et al., ICLR 2022). 우리가 관찰한 "그래프 *내* 노드 간
ego-EPD 는 대부분 퇴화·유사"라는 사실 자체가, 변별이 노드 간이 아니라 **그래프 간**에서 발생함을
시사한다.

**(3) 구조적 이상탐지.** 위상이 class 신호로는 약해도 **outlier 신호**일 수 있다 — 비정상적 ego
위상(비정상 루프 구조)을 갖는 노드 탐지(사기/봇). class-mix 대조가 그대로 이식 가능.

**(4) Featureless/cold-start 노드 표현.** aifb 가 보여준 유일한 진짜 효용. 단 이 경우에도 degree·
PageRank·Laplacian PE 등 **훨씬 싼 구조 인코딩과의 정면 비교가 선행**되어야 한다 — EPD 가 그보다
나은지는 미검증이며, §1.2(iv)의 표현 병목을 감안하면 회의적으로 출발하는 것이 정직하다.

**(5) 메타패스**: 성능 도구가 아니라 **해석·탐색 도구**로 — 발견된 메타패스(관계 합성 가중치)와
homophily 진단을 모델 설명에 쓰는 방향. 성능 목적이면 관계-완전 모델이 기본값.

---

## 5. 추가 발견 (F5–F9)

**F5. Silent failure 와 대조군의 방법론적 가치.** imdb·freebase 의 위상이 **0벡터로 대체된 채**
파이프라인이 조용히 완주했고(§F7), 결과 수치는 "그럴듯한 ≈0 효과"로 보였다. 이를 잡아낸 것은
에러가 아니라 **대조군 간 불일치**(e2 vs f 유의차가 0벡터 가설과 모순)였다. 교훈: (1) "에러 없이
돌았다 ≠ 실험이 수행됐다", (2) noise/mix/manual 대조군은 효과 분해뿐 아니라 **파이프라인 무결성
검증** 역할을 한다. NaN watchdog·특징 분산 assert 같은 런타임 검증을 표준화할 것.

**F6. 백본의 noise 내성 비대칭.** HAN 은 attention 구조 덕에 등방 noise 에 둔감((b1)≈(a)), RGCN 은
유의하게 취약((e1)<(d) 3/7\*). 실무: RGCN 계열에 feature 를 얹을 땐 사전 선별/게이팅 필요.

**F7. GTN 의 고립노드 NaN (원인 규명·수정).** 모든 base relation 에서 고립된 노드(imdb 61개,
freebase 97개; 타 데이터셋 0)가 inter-layer 정규화의 대각 제거 후 zero-row 가 되고,
`torch.where(deg>0, 1/deg, 0)` 의 backward 에서 0×inf=NaN 이 첫 backward 에 GTConv 전체를 오염
(epoch1 재현, 회귀 테스트로 고정). `deg.clamp(min=1e-12)` 로 수정(수치 동일) 후 80 run 재실행,
NaN 0/80. 고립은 subsample(freebase)·kNN 희소화(imdb)가 만든 것 — **전처리가 만든 병리가 모델
수치 안정성을 관통**한 사례.

**F8. 지표 선택이 결론을 뒤집는다.** yelp: macro-F1 은 HAN>RGCN, accuracy 는 RGCN(0.871)≫HAN
(0.624) — 희귀 라벨 포기 vs 다수 정확도의 trade-off. mag: macro-F1 0.02 vs accuracy 0.28(우연의
~100배). 결론 서술에 두 지표 병기가 필수.

**F9. 통계 실무.** 주변 std 가 크게 겹쳐도(aifb ±0.145) paired 검정은 유의를 검출(win 90%,
p=0.020) — seed 짝 없는 mean±std 비교는 검정력 낭비. 반대로 8대조×7데이터셋 다중비교이므로 경계
p(~0.05)는 단독 근거로 쓰지 않았다.

---

## 6. 한계

- 집계 지표(macro-F1/acc)는 within-class 정보에 둔감 — real≈mix 단독으론 "per-node 무의미"의 강한
  단정 불가. 단 **mix>real(유의)·f<d(유의)** 는 방향성 있는 결과로 이 한계의 영향을 받지 않는다.
  per-node 지표(margin/NLL)·η²(위상 유사도 분산비) 진단은 미수행.
- (f) aifb·yelp 미완 — 특히 aifb (f)는 "구조 인코딩 이득이 강한 백본에서도 생존하는가"의 미해결
  질문. class-mix 의 혼합률이 라벨 희소 데이터셋에서 낮음(aifb 7.7%, freebase 9.9%) — 그곳의 mix
  대조는 약함. 백본 비교는 입력 비대칭 포함(§2). 다중비교 보정 미적용.
- b2/e2(acm·dblp·mag·aifb·yelp)와 f(acm·dblp·mag)는 협업 환경 실행값(요약/상세표 경유) — 로컬
  재검산은 imdb·freebase 만 수행.

## 7. 참고문헌

이 세션에서 원문 검증: [V] 표기. 나머지는 표준 문헌(재검증 안 함).

- [V] Yan, Ma, Gao, Tang, Wang, Chen. *Neural Approximation of Graph Topological Features* (PDGNN). NeurIPS 2022, arXiv:2201.12032 — vicinity(지역) EPD·필터(ORC/HKS×2/degree)·~100× 가속·transferability 를 본문에서 확인.
- [V] Lee, Kim, Bu, Yoo, Tang, Shin. *Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective*. ICML 2024, arXiv:2402.04621 — within-class feature shuffle(우리 class-mix 의 1차 출처), A–X 의존성 축소가 성능을 올릴 수 있음.
- Yun et al. *Graph Transformer Networks*. NeurIPS 2019 (GTN).
- Wang et al. *Heterogeneous Graph Attention Network*. WWW 2019 (HAN; semantic attention = 우리 fusion 의 출처).
- Schlichtkrull et al. *Modeling Relational Data with Graph Convolutional Networks*. ESWC 2018 (RGCN; PyG RGCNConv).
- Yan et al. *Link Prediction with Persistent Homology: An Interactive View*. ICML 2021 (TLC-GNN; HKS 필터·PI 워크플로의 원류).
- Adams et al. *Persistence Images*. JMLR 2017.
- Hofer et al. *Deep Learning with Topological Signatures*. NeurIPS 2017 · Carrière et al. *PersLay*. AISTATS 2020 · Horn et al. *Topological Graph Neural Networks (TOGL)*. ICLR 2022 — graph-level PH.
- Strobl et al. *Conditional Variable Importance for Random Forests*. BMC Bioinformatics 2008 (conditional permutation importance — class-mix 의 통계적 원조).
- (오인용 정정) arXiv:2106.04764 는 speaker diarization 논문으로 class-mix 와 무관 — [V] 확인.
- 부속: `paired_stats.md` · `SUMMARY.md` · `CLASS_WISE_MIXING.md` · `f_rgcn_per_seed.md` · `rerun_imdb_freebase.md` · `DATASETS.md`.
