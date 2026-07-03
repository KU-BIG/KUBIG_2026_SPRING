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

## 2. 대조군 기반 핵심 발견

### F1. random noise — 차원 추가 효과 없음 (설계 검증 통과)
(b1)≈(a) 전 데이터셋. RGCN 은 오히려 noise 에 민감하게 하락(e1<d: imdb −0.057, freebase −0.051,
aifb −0.079) — HAN 의 projection/attention 은 무의미한 차원을 걸러내지만 RGCN 은 concat 입력이
관계별 선형변환에 그대로 전파.

### F2. 위상의 aggregate 이득 = "HAN × feature-약함"에서만
(c) vs (b1): dblp +0.086, aifb +0.097. feature 강한 acm/imdb ≈0, mag/yelp ≈0/음수.
**RGCN 위에서는 어떤 위상 조건도 (d)를 넘지 못함.**

### F3. class-mix vs real — per-node 정렬의 기여는 검출되지 않음
(b2)/(e2)는 (c)/(f)와 **동일한 위상 값**을 class·split 안에서만 섞은 것 → real−mix 차이가 per-node
정렬의 기여를 격리. 결과: HAN 에선 real≈mix(dblp 는 mix 가 real 을 이김 0.878>0.862), RGCN 에선
**4/4 전부 mix>real** (dblp +0.080). 위상 가치는 class-level 분포에서 나오고, 정렬은 기여 없음/유해.
"정렬을 깨면 좋아지는" 패턴은 within-class shuffle 이 A–X(구조–특징) 의존성 노이즈를 제거해 성능을
올릴 수 있다는 [Lee et al., ICML 2024](https://arxiv.org/abs/2402.04621) 와 정합.

### F4. 강한 백본(RGCN)에 위상 주입은 손해
(f)<(d) 측정 4/4 (acm −0.022, dblp −0.075, imdb −0.014, freebase −0.043). (e2)≈(d)이므로 원인은
특징 값이 아니라 **정렬된 per-node 위상 + 차원 증가**.

## 3. 데이터셋별 심층 — 위상 이득의 3가지 메커니즘

위상 이득이 있는 곳/없는 곳이 갈리는 이유를 추가 통제(**b2_manual** = HAN 이 이미 쓰는 메타패스
2개 *위에서만* 계산한 위상; 아카이브 10-seed)로 분해했다:

| 데이터셋 | (a) base | b2_manual (HAN 메타패스 위상) | (c) GTN 위상 (전 관계) | 진단 |
|---|---|---|---|---|
| dblp | 0.786 | **0.790 (+0.004)** | 0.862 (+0.076) | 이득은 위상이 아니라 **관계 커버리지** |
| aifb | 0.451 | **0.560 (+0.109)** | 0.575 (+0.124) | 이득은 **구조 인코딩** (featureless 보완) |
| acm | 0.895 | 0.896 (+0.001) | 0.894 (−0.001) | feature 포화 — 아무것도 안 통함 |

**메커니즘 A — 관계 커버리지 누출 (dblp).** dblp 의 클래스(저자 연구분야)를 결정하는 관계는
venue(APVPA)인데, **HAN 메타패스(APA·APTPA)에는 빠져 있고** GTN 채널(전 관계+I)에는 들어간다.
같은 위상 파이프라인을 HAN 의 2개 메타패스 위에서만 돌리면 이득이 **+0.004 로 소멸** → dblp 의
"+0.076 위상 이득"의 실체는 EPD 가 아니라 **GTN 채널을 경유한 누락 관계(venue) 정보의 주입**이다.
mix≈real(class-level 만 중요)·RGCN(APVPA 직접 사용) 0.934 인 것과 모두 정합.

**메커니즘 B — featureless 그래프의 구조 인코딩 (aifb).** aifb 는 one-hot(정보 없음) feature 라
*어떤* 노드별 구조 기술자든 정보가 된다. HAN 과 같은 관계 위에서 계산한 위상만으로도 +0.109 —
즉 여기의 이득은 진짜 "구조를 요약한 노드 특징"으로서의 위상이다. real(0.575)>mix(0.538) 경향도
유일하게 여기서 나타난다(±0.145/±0.106 으로 약함). 단 std 0.145 로 매우 불안정.

**메커니즘 C — feature 포화 (acm·imdb).** 1902~3489 차원의 강한 bag-of-words feature 가 클래스를
거의 결정 → 위상·noise·mix 무엇을 더해도 ≈0. (acm 메타패스 homophily 2.0× 측정 — 구조 신호는
있으나 feature 와 중복.)

**퇴화 케이스 (freebase·mag·yelp).** freebase/yelp 는 featureless 인데도 이득이 없다 — aifb 와의
차이는 규모·구조: freebase(subsample 후 희소·std ±0.06~0.11), yelp(3.9M 엣지 초밀집 + 희귀
멀티라벨), mag(349클래스 subsample → metric 붕괴). 구조 신호가 있어도 지표 분산·라벨 희소성에
묻힌다. → **"featureless 면 위상이 돕는다"가 아니라 "featureless 이면서 구조가 깨끗하고 라벨이
조밀할 때(aifb)"** 가 정확한 조건.

## 4. HAN vs RGCN 심층

accuracy 기준 **RGCN ≥ HAN 7/7** (macro-F1 6/7; yelp 는 지표 착시 — acc 0.871 vs 0.624).

| 요인 | 내용 |
|---|---|
| 입력 커버리지 | RGCN 은 base relation **전부(3–6)**, HAN 은 메타패스 **2개**. dblp 의 격차 +0.148 은 대부분 APVPA(venue) 누락으로 설명(§3-A). **공정성 caveat** — 순수 아키텍처 비교가 아님 |
| 표현력 | RGCN: 관계별 전용 W_r + root self-loop. HAN: 노드 변환 공유 + 메타패스 스칼라 β — 메타패스 2개뿐이라 semantic attention 이 고를 것도 적음 |
| 홈그라운드 | RDF entity 분류(aifb)는 RGCN 원논문 벤치마크 영역 (+0.30) |
| noise 내성 | 역방향: HAN 은 noise 둔감, RGCN 민감(F1) — 그럼에도 RGCN 이 전부 이김 |
| 보조 증거 | dblp 에서 **GTN-only(전 관계 분류기) 0.920 > HAN+위상 0.862** — 병목은 위상이 아니라 HAN+메타패스 부분집합 자체 |

## 5. 메타패스는 노드분류에 적합한 특징인가?

**판정: 사용 가능하지만 관계-완전(relation-complete) 메시지 패싱에 지배당한다(dominated).**

- **찬성 증거**: 메타패스 그래프는 class-homophilous 하다(acm 채널: same-class edge 0.67 vs 무작위
  0.33 = **2.0×**, 시각화 실측). 올바른 메타패스가 있으면 HAN 은 정상 작동.
- **반대 증거**:
  1. **선택(coverage)이 아킬레스건** — dblp 에서 메타패스 2개 선택이 venue 관계를 누락 → RGCN 대비
     −0.148. 메타패스의 성패가 "경로 의미"보다 "어떤 관계를 포함하나"에 좌우됨.
  2. **관계-완전 1-hop(RGCN)이 모든 데이터셋에서 우세** — 경로를 미리 합성할 필요 자체가 없음.
  3. **GTN 으로 메타패스를 학습해도 수동 선택과 비슷** (acm: c2 0.894 vs b2_manual 0.896) — 학습이
     선택 문제를 해결해주지 않음. GTN 의 실질 기여는 경로 발견이 아니라 **전 관계 접근**(§3-A).
  4. 메타패스 사전합성 + kNN 희소화 과정에서 엣지 중복도/가중치 정보 소실.
- **결론**: 메타패스의 가치는 성능이 아니라 **해석성**(어떤 의미 경로가 노드를 묶는지 §시각화)이다.
  성능 목적이면 관계를 전부 주는 모델(RGCN 계열)이 우선.

## 6. Robustness (seed 분산) — 위상 추가로 안정성이 올라가지 않는다

| 데이터셋 | (a) std | (c) std | (d) std | (f) std |
|---|---|---|---|---|
| acm | 0.006 | 0.008 | 0.006 | 0.006 |
| dblp | 0.013 | 0.013 | 0.004 | **0.017 ↑** |
| imdb | 0.009 | **0.019 ↑** | 0.004 | 0.004 |
| freebase | 0.055 | 0.051 | 0.108 | 0.065 ↓ |
| mag | 0.012 | 0.007 ↓ | 0.053 | – |
| aifb | 0.040 | **0.145 ↑↑** | 0.018 | – |
| yelp | 0.028 | 0.024 | 0.006 | – |

개선 1~2 / 악화 2~3 / 나머지 ≈. 특히 **aifb std 3.6×**, imdb 는 accuracy std 0.002→0.110(일부
seed 학습 붕괴) — 위상 concat 이 오히려 seed 민감성을 키우는 경우가 있다. 초기 3-seed 에서 보였던
"위상 변형의 분산 감소" 힌트는 **10-seed 에서 재현되지 않았다.** (여기서의 robustness = seed 분산.
입력 교란/적대적 내성은 미실험 — 주장 불가.)

## 7. 향후 과제 — 위상 특징 × link prediction

**가설이 성립할 근거**: (i) 본 결과의 실패 지점은 "per-node 정렬"인데, LP 는 **노드쌍(pair) 단위
지역 구조**가 직접적 신호인 과제다 — EPD 가 포착하는 **루프(H1)는 링크 형성(triadic closure)과
기계적으로 연결**된다. (ii) 문헌도 LP 쪽이 우호적: TLC-GNN(본 파이프라인의 원류)은 애초에 PH 기반
**LP 방법**이고, PDGNN 논문도 LP 벤치마크에서 true-EPD 동급 성능을 보고.

**정직한 반대 근거**: 본 프로젝트의 선행 hetero-LP 실험(hetero_pdg_lp)에서는 **통제 후 위상 신호가
null** 이었다(6설정 일관). 따라서 재도전한다면 본 분석의 교훈을 설계에 반영해야 한다:
feature-약한/featureless 세팅 우선(§3-B), **pair-level permutation 대조군**(본 연구의 class-mix 에
대응하는 "같은 거리/공통이웃 bucket 내 셔플"), per-pair metric(AUC 외 hard-pair subset), 관계
커버리지 통제(§3-A 의 누출 방지). 이 통제 없이 LP 이득을 보고하면 본 연구가 NC 에서 밝힌 것과
같은 confound(커버리지·class-level 누출)를 반복할 위험이 크다.

## 8. 한계

- **Aggregate metric 둔감성**: real≈mix 는 per-node metric(margin/NLL) 없인 강한 단정 불가
  (거의 tautology). 반면 **mix>real(RGCN 4/4)은 방향성 있는 검출된 결과**. η²(within/between-class
  위상 유사도) 진단 미수행.
- (f) 4/7 부분 완료. b2/e2/f per-run 원본은 협업 환경에 있어 paired 검정 재계산 생략.
- 백본 비교는 입력 비대칭(관계 3–6 vs 메타패스 2) 하의 "표준 사용법" 비교 — §3-A 가 그 영향 크기를
  dblp 에서 정량화(+0.07~0.15 상당).
- b2_manual 수치는 이전 캠페인 아카이브(동일 seed 10개, 동일 프로토콜)에서 가져옴.

## 9. 참고

- Yun et al., *Graph Transformer Networks*, NeurIPS 2019 (GTN)
- Yan et al., *Neural Approximation of Graph Topological Features*, NeurIPS 2022 (PDGNN)
- Wang et al., *Heterogeneous Graph Attention Network*, WWW 2019 (HAN)
- Schlichtkrull et al., *Modeling Relational Data with GCNs*, ESWC 2018 (RGCN)
- Yan et al., *Link Prediction with Persistent Homology: An Interactive View*, ICML 2021 (TLC-GNN)
- Lee et al., *Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution*, ICML 2024
- Strobl et al., *Conditional variable importance for random forests*, BMC Bioinformatics 2008
