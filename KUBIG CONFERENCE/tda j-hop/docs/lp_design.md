# LP 확장 설계 — 위상 특징 × 이종 Link Prediction

> NC 캠페인의 후속. 질문: **per-node 수준에서 무기여였던 EPD 위상이, pair(엣지) 수준 과제에서는
> 기여하는가?** 문헌(TLC-GNN ICML'21, PDGNN NeurIPS'22 Table 3)은 LP 에 우호적이나, 본 프로젝트
> 선행 hetero-LP 는 통제 시 null — 이번 수준의 대조군으로 재검증한 적이 없다.

## 1. 범위 (시간 제약 반영)

- **데이터셋 3개 (소형·고속)**: aifb(2,270·featureless·위상 최선케이스) / acm(3,025·feature 포화
  대조) / dblp(4,057·관계누출 케이스). 각각 NC 에서 규명한 메커니즘 B/C/A 를 대표.
- **예측 대상**: 주 타겟–타겟 base relation 의 엣지 (acm=PAP, dblp=APA, aifb=r0) — 기존
  HeteroBundle·파이프라인 최대 재사용. 표준 벤치마크(HGB-LP)와 다른 합성엣지 예측임을 명시.
- **엣지 샘플링**: positive ≤5,000개 샘플(전체가 그보다 작으면 전부), train/val/test = 80/10/10,
  negative 1:1 uniform (train 은 매 epoch 재샘플, val/test 는 고정). **test 엣지는 인접행렬에서
  제거**(leakage 방지 — 메시지패싱·위상 계산 모두 train 엣지만 사용).
- **seed 10개** (기존 `experiments/seeds.txt` 동일 — NC 와 비교 일관성).

## 2. 2단계 실험

### Level 1 — node-level PI 스캔 (기존 파이프라인 재사용, 빠름)
인코더가 [x ⊕ 노드 PI] 를 받는 구조 그대로, task 만 LP 로:
`score(u,v) = MLP([z_u ⊙ z_v, z_u, z_v])`, BCE, 평가 AUC-ROC(+hits@k 보조).

| 조건 | 내용 | NC 대응 |
|---|---|---|
| L1-a | encoder 단독 (위상 없음) | (a)/(d) |
| L1-b | + random noise (동일 차원) | (b1)/(e1) |
| L1-c | + node PI (GTN-PDGNN, real) | (c)/(f) |
| L1-m | + node PI **pair-safe shuffle** | (b2)/(e2) 대응 |

encoder = RGCN (NC 승자, 관계-완전). 3 ds × 4 조건 × 10 seed = **120 run** (소형이라 run당 10–20분).

### Level 2 — pair-vicinity EPD 본실험 (문헌 방식, PDGNN 의 존재이유가 실제로 작동하는 지점)
후보쌍 (u,v)마다 **vicinity 그래프 = ego_k(u) ∪ ego_k(v)** (train 엣지만, k=1)에서 EPD → PI.
정확 EPD(gudhi)는 학습 라벨 샘플에만, 전 후보쌍은 PDGNN 추론 (TLC-GNN/PDGNN 충실).
`score(u,v) = MLP([z_u ⊙ z_v, PI_pair(u,v)])`.

| 조건 | 내용 |
|---|---|
| L2-real | pair PI 그대로 |
| L2-mix | pair PI 를 **공통이웃수(CN) bucket 내 셔플** — NC class-mix 의 LP 판: "쉬운 구조 공변량(CN)을 넘는 pair-위상 정보가 있는가" 격리 |
| L2-noise | pair 슬롯에 random noise |
| (기준) | L1-a (pair 특징 없음) |

Level 1 에서 신호 유무와 무관하게 **aifb + 1개 이상**에서 수행 (Level1 null 이어도 pair-level 은
별개 질문). 3 ds × 3 조건 × 10 seed = 90 run (pair EPD 계산 포함 run당 ~30–60분).

## 3. 대조군 논리 (NC 에서 배운 것 이식)

- **noise**: 차원 추가 효과 격리 (NC F1(i): RGCN 은 noise 만으로 하락했음 — LP head 에서도 확인).
- **mix**: LP 에는 class 가 없으므로 **구조 공변량 bucket**(공통이웃수, 필요시 Adamic-Adar 분위)
  내 셔플 — "CN 으로 설명되는 것 이상"을 격리. cf. conditional permutation (Strobl 2008),
  within-group shuffle (Lee et al. ICML 2024).
- **관계 누출 통제** (NC §3.2-A 교훈): GTN 채널이 예측 대상 관계 *자체*를 합성에 쓰면 자명한 누출
  → **위상 계산용 채널 합성에서 예측 대상 relation 을 제외**한 변형(L1-c′)을 dblp 에서 추가 확인.
- **판정표**: real>mix>noise ⇒ pair-위상 고유 신호 / real≈mix>noise ⇒ CN 수준 정보뿐 /
  전부≈기준 ⇒ LP 에서도 null (선행 hetero-LP null 재확인).

## 4. 구현 계획 (전부 CPU 개발 가능, GPU 는 캠페인만)

1. `tda/lp.py` — 엣지 분할(+인접 제거)·negative sampling·LP head·AUC 평가·조건 플래그.
2. `tda/topology/pair_epd.py` — pair vicinity 구성 + PDGNN 학습/추론 (epd.py 재사용).
3. toy 데이터 CPU 검증 + 회귀 테스트.
4. `experiments/gen_lp.py` + 캠페인 스크립트 (%N throttle, 자동 집계) — 클러스터 한산 시 발사.

## 5. 예상 규모

Level1 120 + Level2 90 = **210 run**, 소형 데이터셋이라 총 ~60–90 GPU·h (7장 병렬 시 wall ~9–13h).
Level1 먼저 발사 → 결과 보고 Level2 범위 조정 가능.
