# Hetero-PDGNN on DBLP — 실험 로그

기간: 코드 작성 + Colab 실행 ~1세션
프레임워크: PyG + TLC-GNN (dionysus→gudhi 패치) + 자체 wrapper
데이터셋: DBLP (PyG 표준, 4 node types, 4-class author classification)

---

## 1. 연구 가설 및 동기

### 1.1 출발점
- **PDGNN (NeurIPS 2022, Yan et al.)**: homogeneous graph에서 Extended Persistence Diagram (EPD)을 GNN의 임베딩에 보조 feature로 주입하면 node/link 성능 향상.
- 기존 PDGNN 논문은 **homogeneous graph**만 다룸 (Cora, Citeseer 등). Heterogeneous graph에 대한 확장 미수행.

### 1.2 핵심 가설
> meta-path를 통해 hetero graph를 homogeneous subgraph로 분해하면, 각 subgraph에서 PDGNN/EPD 추출이 자연스럽게 가능. 이 EPD feature를 hetero GNN(HAN) 임베딩에 concat하면 homogeneous case처럼 성능 개선이 일어날 것.

### 1.3 핵심 도전 (CS224W 강의 09 + PDGNN 구현 가이드)
1. 노드 타입별 feature space가 달라 filter function 정의 모호 → meta-path subgraph 사용으로 우회
2. 타입 간 filter value 비교 무의미 → meta-path subgraph는 homogeneous라 자동 해소
3. Union-Find 비교 연산 — homogeneous subgraph에서는 수정 불필요
4. GNN backbone — HAN을 그대로 사용

이 중 **(1), (2), (3)을 meta-path 접근으로 일거에 해소**하는 것이 본 실험의 핵심 설계 선택.

---

## 2. 구현 (`hetero_pdgnn/` 패키지)

```
hetero_pdgnn/
├── data_metapath.py    DBLP 로드 + meta-path 인접행렬 곱 + NX subgraph
├── epd_features.py     노드별 EPD persistence-image 추출 (캐싱)
├── models.py           HAN baseline + HAN+EPD
├── train.py            학습/평가/seed별 비교
└── run_colab.py        Colab 오케스트레이션 (7개 셀)
```

### 2.1 Meta-path 선택 (저자 분류용 표준 3종)
- **APA**: author–paper–author (공저)
- **APCPA**: author–paper–conference–paper–author (같은 학회)
- **APTPA**: author–paper–term–paper–author (같은 주제)

각 meta-path를 인접행렬 곱으로 계산: $A_1 \cdot A_2 \cdot \ldots$ → (author × author) sparse matrix → 대칭화 + binarize → NetworkX undirected graph.

### 2.2 EPD feature 추출 파이프라인 (per author v, per meta-path)
1. v의 k-hop ego-subgraph 추출
2. Filter function 계산 (`distance` 기본): v로부터 normalized shortest-path distance
3. gudhi `SimplexTree.extend_filtration()` → 0D ordinary + 1D extended diagram
4. Persistence imager → R×R image → flatten
5. 3개 meta-path EPD image를 concat

최종 feature shape: `[num_authors, num_metapaths × R²]`

### 2.3 모델 구조
- **HAN baseline**: PyG `HANConv` (meta-path 엣지만 사용) → linear classifier
- **HAN+EPD**: HAN 출력 임베딩과 projected EPD를 concat → classifier
  - EPD projection: 2-layer MLP (epd_dim → 64 → 64)

---

## 3. 인프라 단계 (실험 전 디버깅)

### 3.1 Colab 셋업
- TLC-GNN 레포 clone, dionysus → gudhi 패치 적용
- DBLP는 PyG가 자동 다운로드
- 업로드 트러블 (file panel sync 이슈) → ZIP으로 일괄 업로드 해결

### 3.2 Meta-path 밀도 문제 발견

binarize만 한 meta-path subgraph 크기:

| meta-path | nnz (raw count) | edges (binarize) | avg deg | density |
|---|---:|---:|---:|---:|
| APA   | 11,113 | 3,528 | 1.7 | 0.04% |
| APCPA | 5,000,495 | 2,498,219 | 1,230 | 30% |
| APTPA | 7,043,571 | 3,519,757 | 1,735 | 43% |

**원인**: 학회(20개), 용어(7723개)가 제한적이라 카운트가 heavy-tailed 분포 (APCPA: median=2, p99=61, max=4124). binarize는 약한 연결(우연한 공유 1~2회)과 강한 연결(같은 그룹)을 똑같이 취급 → 그래프가 거의 complete graph 수준으로 폭발.

### 3.3 해결 — k-NN sparsification (k=20)
각 노드에서 카운트 상위 20명 이웃만 유지 → 대칭화 + binarize.

| meta-path | edges (k-NN, k=20) | avg deg |
|---|---:|---:|
| APA   | 3,524 | 1.7 (이미 sparse) |
| APCPA | 80,264 | 39.6 |
| APTPA | 80,486 | 39.7 |

EPD 계산 시간이 합리적 수준(분 단위)으로 축소.

### 3.4 hop 튜닝
- hop=2 + k-NN sparsified: APA는 OK, APCPA는 첫 500노드 milestone 안 뜨고 정체 (2-hop ego ~1600노드 폭발)
- **hop=1로 후퇴**: ego ~40노드, APCPA/APTPA 각 ~50초로 완료, 총 96초

→ 최종 EPD config: **k-NN(k=20) + hop=1 + resolution=5** (75-dim per author)

---

## 4. 본격 실험 — HAN vs HAN+EPD

### 4.1 Experiment 1 — 첫 시도 (정보 비대칭 상태)
- HAN: **dense** meta-path edges (12M edges)에서 학습
- EPD: **k-NN sparsified** subgraph에서 계산
- 5 seeds

| Model | Test F1 mean | std | wins |
|---|---:|---:|---:|
| HAN | **0.9205** | 0.0037 | 5/5 |
| HAN+EPD | 0.9172 | 0.0027 | 0/5 |

**판정**: HAN+EPD가 떨어짐. 단, HAN이 훨씬 많은 엣지를 본 비대칭 → 비교 불공정.

### 4.2 Experiment 2 — 정보 대칭 (양쪽 모두 sparsified subgraph)
- HAN과 EPD 모두 k-NN sparsified subgraph 사용
- 5 seeds

| Model | Test F1 mean | std | wins |
|---|---:|---:|---:|
| HAN | 0.9326 | 0.0047 | 2/5 |
| HAN+EPD | **0.9347** | **0.0015** | 3/5 |

**판정**: 역전. HAN+EPD가 mean +0.002, std 1/3 수준. 단 5 seed로는 통계적 유의 불충분.

부가 발견: **두 모델 모두 dense → sparsified로 가면서 향상** (0.92 → 0.93). sparsification이 noise 제거 효과까지 줌.

### 4.3 Experiment 3 — 10 seed로 확장
| Model | Test F1 mean | std | wins |
|---|---:|---:|---:|
| HAN | 0.9332 | 0.0037 | 4/10 |
| HAN+EPD | **0.9344** | **0.0020** | 6/10 |

per-seed difference (HAN+EPD − HAN):
```
seed 0: -0.0033   seed 5: +0.0009
seed 1: -0.0011   seed 6: +0.0028
seed 2: -0.0009   seed 7: +0.0037
seed 3: +0.0014   seed 8: -0.0059
seed 4: +0.0128 ← outlier   seed 9: +0.0018
```

**Paired t-test**: t ≈ 0.77, **p > 0.05**, mean 차이는 통계적 유의 X.

**seed 4 분석**: HAN이 단일 seed에서 0.9235로 떨어짐 (다른 seed 0.93+). 이 outlier가 mean 차이의 주된 원인. seed 4 제외 시 두 모델 거의 동률.

**결론**: HAN+EPD가 outlier seed를 막아주는 robustness 효과 확인. peak 성능은 동률.

### 4.4 Experiment 4 — Resolution=10 (EPD 표현력 증대 시도)
가설: 75-dim이 작아서 mean 효과가 약할 수 있음. 300-dim으로 늘리면 개선?

| Model | Test F1 mean | std |
|---|---:|---:|
| HAN | 0.9332 | 0.0037 |
| HAN+EPD (res=10) | 0.9338 | 0.0024 |

(참고: res=5는 0.9344)

**판정**: 오히려 약간 후퇴. 가능 원인:
- Overfit (train set 400 author 대비 EPD 300 dim 추가는 과다)
- PI 희소성 (hop=1 ego의 EPD point가 적은데 10×10 grid에 분산 → 셀 대부분 ~0)
- MLP bottleneck (300→64 압축에서 정보 손실)

**res=5 유지가 최적.**

### 4.5 Experiment 5 — Degree filter (source 무관한 글로벌 filter)
가설: distance filter는 v 종속이라 ego마다 의미 달라짐. degree filter는 글로벌이라 더 안정적일 수 있음. PDGNN 원논문도 degree 기반 사용.

| Model | Test F1 mean | std | wins |
|---|---:|---:|---:|
| HAN | 0.9332 | 0.0037 | 9/10 |
| HAN+EPD (degree) | 0.9283 | 0.0020 | 1/10 |

**판정**: Mean 큰 폭 하락 (-0.0049). 하지만 **std는 여전히 0.0020 유지**.

해석:
- distance filter는 **node-centric** ("v의 관점에서 본 위상")
- degree filter는 **structure-centric** ("ego의 구조 모양")
- 노드 분류에서는 node-centric이 명백히 우수

---

## 5. 종합 결과

### 5.1 모든 실험 요약 (10 seeds 기준)

| 설정 | mean F1 | std | seeds won (vs HAN) | Δ mean |
|---|---:|---:|---:|---:|
| HAN baseline | 0.9332 | 0.0037 | — | — |
| HAN+EPD (distance, res=5) | **0.9344** | **0.0020** | 6/10 | **+0.0012** |
| HAN+EPD (distance, res=10) | 0.9338 | 0.0024 | 4/10 | +0.0006 |
| HAN+EPD (degree, res=5) | 0.9283 | 0.0020 | 1/10 | −0.0049 |

### 5.2 일관된 발견 vs 가변적 발견

**일관된 발견 (filter/resolution 무관)**:
- HAN+EPD의 **std는 항상 ~0.0020**으로 수렴 (baseline 대비 ~50% 감소)
- HAN의 outlier seed (0.9235)를 모든 EPD variant가 막아줌
- → EPD가 어떤 정보든 추가되면 학습이 **regularize / stabilize** 됨

**가변적 발견 (filter 선택에 민감)**:
- Mean 개선 여부는 filter에 매우 민감
- distance > degree (큰 차이)
- resolution은 5가 10보다 약간 우수

### 5.3 통계적 신중함

5 seed → 10 seed로 늘리면서 mean 차이가 +0.0021 → +0.0012로 줄어듦. 차이가 seed 4 outlier에 강하게 의존하는 패턴 확인. **peak performance gain은 통계적으로 unconfirmed**.

반면 std 감소는 모든 실험에서 robust하게 재현 → 이게 진짜 신호.

---

## 6. 잠정 결론

### 6.1 가설 부분 검증
원 가설(EPD가 hetero에서도 mean F1을 의미 있게 높일 것)은 **약하게 지지**됨. mean 차이는 작고 통계적 유의 미달.

### 6.2 새 발견
EPD의 진짜 기여가 **peak가 아니라 robustness**일 수 있음.
- std ~50% 감소 (일관됨)
- bad initialization으로 인한 outlier seed 방지
- 이는 PDGNN 원논문이 강조한 효과(성능 향상)와는 다른 결의 기여

### 6.3 DBLP 특이성 가능성
- 4-class author classification은 ~0.93에서 saturated 양상
- 더 difficult한 데이터셋에서는 peak gain도 의미 있을 수 있음

---

## 7. 다음 방향 후보

### (1) 다른 hetero 데이터셋 검증 — 권장 1순위
- ACM, IMDB (PyG 표준)에 동일 분석 적용
- robustness gain이 데이터셋 독립 현상인지 확인
- 코드 거의 그대로, dataset 로더만 교체. 추가 작업량 작음

### (2) Idea 2 — Type-aware learnable filter
- 원래 계획의 두 번째 아이디어 (CS224W 노트의 type-conditioned MLP filter)
- meta-path subgraph 우회 → hetero graph 전체에서 type-aware EPD
- 본질적으로 다른 접근이므로 새 신호 가능성
- 구현 복잡도 중상

### (3) Robustness 가설 심화 분석
- training loss curve 비교 (EPD 추가 시 더 smooth한가?)
- train mask 크기 축소(200/100/50)에서도 std 감소 유지되는지
- 다른 regularization (dropout↑, weight decay↑)으로 비슷한 std 감소 만들 수 있는지
- → EPD가 어떤 regularization 효과를 주는지 mechanism 분석

### (4) hop=2 + k=10 (B-β, 미시도)
- avg_deg 20, hop=2 ego ~400 노드
- EPD 더 풍부하게 받아 peak 끌어올릴 수 있는지 확인
- 시간 비용: EPD 재계산 ~5분 + 학습 2분

### 권장 진행 순서

**(1) → (2) → (3)**

- (1)은 가장 쉽고 일반성 강화에 핵심
- (2)는 본 연구의 새로운 기여 영역 — DBLP에서 결과가 약했어도 type-aware 접근으로 mean이 살아날 가능성
- (3)은 (1)에서 robustness가 일반화되면 자연스럽게 따라가는 분석

(4)는 우선순위 낮음 — filter 변경에서 본 효과 크기로 미루어 큰 mean 변화 기대 어려움

---

## 8. 코드 / 데이터 산출물

- 모든 EPD feature: `/content/epd_cache/epd_{APA,APCPA,APTPA}_*.npy` (Colab 세션 종료시 휘발)
- 모델 가중치는 저장 안 함 (5/10 seed 비교 목적이라 불필요)
- 그래프 fingerprint 기반 캐싱이므로 sparsification 파라미터 바뀌면 자동 재계산

---

## 9. 메모

- **DBLP는 4-class author classification 기준 ~93%에서 모델 간 큰 차이가 나기 어려운 saturated 영역**일 가능성
- HAN 자체가 meta-path attention 두 단계를 거치므로 위상적 정보를 이미 어느 정도 학습 → EPD의 추가 mean gain이 작은 게 자연스러움
- robustness gain (std ↓)이 진짜 contribution이라면 이를 paper의 main story로 세팅하는 것이 더 정직하고 흥미로움
