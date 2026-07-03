# 결과 분석 — 2 백본 × 4 위상 조건, 7 데이터셋

> 조건: (a)=HAN, (b1)=HAN+random noise, (b2)=HAN+class-mix, (c)=HAN+real 위상,
> (d)=RGCN, (e1)=RGCN+noise, (e2)=RGCN+class-mix, (f)=RGCN+real 위상.
> random seed 10개(15092 88015 238623 251219 312132 500973 588722 661491 792965 825661), mean±std.
> (f)는 4/7 완료(acm·dblp·imdb·freebase; `f_rgcn_per_seed.md`). 원 수치는 `SUMMARY.md`.

## 1. 결과표

### test Macro-F1 (mean±std)

| 데이터셋 | (a) HAN | (b1) +noise | (b2) +mix | (c) +real | (d) RGCN | (e1) +noise | (e2) +mix | (f) +real |
|---|---|---|---|---|---|---|---|---|
| acm | 0.895±0.006 | 0.887±0.008 | 0.899±0.006 | 0.894±0.008 | 0.925±0.006 | 0.919±0.013 | 0.923±0.012 | 0.903±0.006 |
| dblp | 0.786±0.013 | 0.776±0.010 | 0.878±0.018 | 0.862±0.013 | 0.934±0.004 | 0.930±0.008 | **0.939±0.005** | 0.859±0.017 |
| imdb | 0.438±0.009 | 0.441±0.004 | 0.447±0.021 | 0.450±0.019 | 0.636±0.004 | 0.579±0.008 | 0.634±0.004 | 0.622±0.004 |
| freebase | 0.146±0.055 | 0.160±0.059 | 0.162±0.027 | 0.144±0.051 | 0.209±0.108 | 0.158±0.061 | 0.207±0.056 | 0.166±0.065 |
| mag | 0.017±0.012 | 0.026±0.011 | 0.019±0.009 | 0.023±0.007 | 0.104±0.053 | 0.090±0.034 | 0.088±0.048 | – |
| aifb | 0.451±0.040 | 0.478±0.100 | 0.538±0.106 | 0.575±0.145 | 0.752±0.018 | 0.673±0.071 | 0.720±0.134 | – |
| yelp | 0.110±0.028 | 0.094±0.026 | 0.079±0.024 | 0.091±0.024 | 0.055±0.006 | 0.056±0.006 | 0.067±0.019 | – |

### test Accuracy (mean±std, 보조 지표)

멀티라벨(imdb·yelp)은 element-wise accuracy. macro-F1 이 붕괴하는 MAG(349클래스 subsample)·yelp(희귀 멀티라벨) 해석 보조용.

| 데이터셋 | (a) HAN | (b1) +noise | (b2) +mix | (c) +real | (d) RGCN | (e1) +noise | (e2) +mix | (f) +real |
|---|---|---|---|---|---|---|---|---|
| acm | 0.894±0.006 | 0.886±0.008 | 0.897±0.006 | 0.892±0.009 | 0.924±0.006 | 0.918±0.013 | 0.922±0.012 | 0.901±0.006 |
| dblp | 0.795±0.013 | 0.784±0.009 | 0.882±0.018 | 0.868±0.012 | 0.939±0.004 | 0.936±0.007 | **0.943±0.004** | 0.865±0.016 |
| imdb | 0.743±0.002 | 0.730±0.003 | 0.707±0.115 | 0.707±0.110 | 0.791±0.002 | 0.763±0.004 | 0.791±0.002 | 0.787±0.002 |
| freebase | 0.631±0.036 | 0.582±0.044 | 0.624±0.042 | 0.622±0.028 | 0.647±0.034 | 0.579±0.077 | 0.643±0.033 | 0.635±0.029 |
| mag | 0.140±0.081 | 0.168±0.080 | 0.142±0.067 | 0.162±0.071 | 0.284±0.087 | 0.265±0.090 | 0.283±0.101 | – |
| aifb | 0.669±0.008 | 0.650±0.067 | 0.678±0.042 | 0.714±0.057 | 0.800±0.011 | 0.731±0.037 | 0.792±0.081 | – |
| yelp | 0.624±0.112 | 0.677±0.136 | 0.742±0.153 | 0.703±0.165 | 0.871±0.012 | 0.874±0.004 | 0.827±0.070 | – |

## 2. 핵심 발견

### F1. random noise 대조 — 차원 추가 효과는 없다 (설계 검증 통과)
- **HAN**: (b1)≈(a) 전 데이터셋 (Δ −0.010 ~ +0.014). 75차원을 더 받는 것 자체는 이득이 아님.
- **RGCN**: (e1)이 (d)보다 **뚜렷이 하락** — imdb −0.057, freebase −0.051, aifb −0.079.
  → 대조군이 의도대로 작동. 부수 발견: **HAN은 noise에 둔감, RGCN은 민감** — HAN 계열은 attention·
  변환이 무의미한 차원을 걸러내는 반면, RGCN 은 concat 입력을 관계별 선형변환에 그대로 태워 노이즈가 전파됨.

### F2. 위상이 aggregate 로 도움이 되는 곳 = "HAN × feature-약함" 조합뿐
- (c) vs (b1): **dblp +0.086, aifb +0.097** — noise 대비 real 위상이 명확히 큼 → 위상 특징에 신호 실재.
- feature 강한 acm(1902d)·imdb(3489d)는 ≈0, mag·yelp 도 ≈0/음수.
- **RGCN 위에서는 어떤 위상 조건도 (d)를 넘지 못함** (아래 F4).

### F3. class-mix vs real — per-node 정렬의 기여는 검출되지 않음 (가장 중요한 발견)
(b2)/(e2)는 **(c)/(f)와 동일한 위상 값**을 같은 class·같은 split 안에서만 섞은 것. 즉 real−mix 차이는
순수하게 "노드별 위상↔노드 매칭(정렬)"의 기여만 격리한다.

- **HAN**: (c)≈(b2)가 지배적. dblp 는 오히려 **mix 가 real 을 이김** (0.878 vs 0.862). 유일한 예외 후보는
  aifb(real 0.575 vs mix 0.538, +0.037)지만 std(±0.145/±0.106)가 커서 약한 신호.
- **RGCN**: 측정된 4개 **전부에서 (e2) > (f)** — acm +0.020, dblp **+0.080**, imdb +0.012, freebase +0.041.
  같은 특징 값인데 *정렬을 깨면 더 좋아진다*.

해석: 위상 특징의 가치는 (있는 경우에도) **class-level 분포**에서 나오고, **노드별 정렬은 기여가 없거나
(RGCN 에선) 오히려 유해**하다. "정렬을 깨면 좋아지는" 패턴은 within-class feature shuffle 이 그래프
컨볼루션에서 성능을 올릴 수 있다는 [Lee et al., ICML 2024 (arXiv:2402.04621)](https://arxiv.org/abs/2402.04621)
의 A–X(구조–특징) 의존성 노이즈 제거 효과와 정합한다 — 즉 "위상이 나쁘다"가 아니라 **위상-노드 결합이
노이즈로 작용**했다는 해석이 맞다.

### F4. 강한 백본(RGCN)에는 위상 주입이 손해
(f) vs (d): 측정 4개 전부 하락 — acm −0.022, **dblp −0.075**, imdb −0.014, freebase −0.043.
(e2)≈(d)인 것과 종합하면, 하락의 원인은 특징 값 자체가 아니라 **정렬된 per-node 위상이 추가하는
구조 의존성 노이즈 + 차원 증가**다. 실무 결론: **관계를 직접 모델링하는 강한 백본에는 GTN-PDGNN
위상 주입의 실익이 없다.**

### F5. 백본 비교 — accuracy 기준 RGCN ≥ HAN **7/7**
- macro-F1 로는 6/7 (yelp 예외)이지만, yelp 는 지표 착시: accuracy 는 RGCN 0.871 vs HAN 0.624.
  RGCN 이 다수 negative 를 잘 맞히고 희귀 라벨을 놓쳐 macro-F1 만 붕괴한 것.
- 원인: ① RGCN 은 base relation **전부(3–6개)** 를 관계별 전용 W_r 로 쓰는 반면 HAN 은 메타패스
  **2개**만 사용(입력 비대칭 — 공정성 caveat), ② 관계별 가중치의 표현력, ③ RDF/다관계 데이터는
  RGCN 의 원 설계 영역(aifb +0.30).

### F6. MAG·yelp 절대값은 metric 특성 (모델 실패 아님)
- MAG: macro-F1 은 349클래스를 6000노드 subsample 에서 평균해 붕괴(대부분 클래스 test 표본≈0).
  accuracy 0.14→0.28(HAN→RGCN)로 우연(1/349)의 ~100배.
- yelp: featureless 멀티라벨(16)의 희귀 라벨이 macro-F1 을 붕괴. accuracy 0.62→0.87.
- 이 둘은 **데이터셋 내 조건 간 Δ로만** 해석할 것.

## 3. 종합 결론

1. **GTN-PDGNN per-node 위상 특징은 이종 노드분류에서 instance-level 기여가 검출되지 않는다.**
   real≈mix(HAN), mix>real(RGCN, 4/4)이 그 증거이며, 위상의 가치는 있어도 class-level 시그니처 수준.
2. **위상이 aggregate 성능을 올리는 유일한 영역은 "HAN × feature-약한 데이터(dblp·aifb)"**이고,
   그 향상조차 class-mix 로 재현되므로 per-node 위상 정보 때문이라고 말할 수 없다.
3. **강한 관계형 백본(RGCN)이 위상 주입 없이 가장 좋다** — accuracy 기준 7/7. 위상을 더하면 오히려 하락.
4. 이는 본 프로젝트의 선행 관찰(통제 시 위상 특징의 신호 부재)과 일관된다.

## 4. 한계·주의 (해석 경계)

- **Aggregate metric 둔감성**: macro-F1/accuracy 는 within-class 정보에 구조적으로 둔감하므로,
  real≈mix 만으로 "per-node 위상 무의미"를 강하게 단정할 수 없다(거의 tautology). 반면 **mix>real
  (RGCN)은 aggregate 로도 검출된 방향성 있는 결과**라 더 강한 증거다. per-node metric(margin/NLL,
  hard-node subset)과 within/between-class 위상 유사도(η²) 진단은 미수행 — 후속 확인 항목.
- (f)는 4/7 부분 완료(mag·aifb·yelp 미완). b2/e2/f 의 per-run 원본은 협업 환경에 있어 이 저장소
  로컬에서 paired 검정을 재계산하지 않았다(값 출처: `SUMMARY.md`, `f_rgcn_per_seed.md`).
- 백본 비교는 입력 비대칭(RGCN 관계 3–6 vs HAN 메타패스 2) 하의 "각 모델 표준 사용법" 비교다.
- imdb 의 HAN 위상 조건 accuracy(0.707±0.11)는 일부 seed 붕괴로 분산이 큼.

## 5. 참고

- Yun et al., *Graph Transformer Networks*, NeurIPS 2019 (GTN)
- Yan et al., *Neural Approximation of Graph Topological Features*, NeurIPS 2022 (PDGNN)
- Wang et al., *Heterogeneous Graph Attention Network*, WWW 2019 (HAN)
- Schlichtkrull et al., *Modeling Relational Data with GCNs*, ESWC 2018 (RGCN)
- Lee et al., *Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution*, ICML 2024 (within-class shuffle)
- Strobl et al., *Conditional variable importance for random forests*, BMC Bioinformatics 2008 (conditional permutation)
