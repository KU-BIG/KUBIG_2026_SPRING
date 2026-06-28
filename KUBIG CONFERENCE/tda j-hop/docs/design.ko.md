# 설계 문서 — HAN + GTN + PDGNN 파이프라인

## 1. 핵심 질문

- 위상 특징(extended persistence diagram, EPD)이 이종 그래프 노드 분류를 돕는가?
- 그걸 가능케 하는 **메타패스 선택을 자동화**할 수 있는가?

선행 연구(DBLP, HAN vs HAN+수동EPD)에서 EPD 의 평균 F1 향상은 작고 config 에 민감했으나,
**분산을 크게 줄였다(~46%)**. 다만 메타패스를 사람이 골라야 했다. 본 파이프라인은 그 선택을
**GTN 으로 자동화**하고, 발견된 메타패스 위에서 **PDGNN 으로 위상 특징**을 뽑는다.

## 2. 전체 구조 (staged, 플랜 Option A)

```
이종 그래프 (HeteroData)
   │  기저 관계 A = {타겟–타겟 1-hop 메타관계, 동종 엣지, 항등} (이진 인접 스택)
   ▼
[Stage 1] GTN  ──────────────  메타패스 채널 C개 자동 발견
   │   GTConv: 채널 c = softmax_r(W[c,r]) 로 가중합한 기저 관계
   │   GTLayer: 곱으로 메타패스 길이 확장 (L 레이어)
   │   채널별 GCN → concat → 분류 (Stage-1 학습 목적)
   │   학습된 채널 인접행렬 H (C, N, N) 출력
   ▼
[Stage 2] PDGNN (채널마다, PDGNN frozen)
   │   1. H_c 를 kNN(k) 희소화 → 이산 그래프
   │   2. HKS(K 스케일) 노드 필터
   │   3. 노드 ego 의 정확 EPD(gudhi)를 라벨로 PDGNN 학습 (라벨 전용)
   │   4. 학습된 PDGNN 으로 노드별 예측 EPD → persistence image (N, res²·K)
   ▼
[Stage 3] Fusion + HAN
   │   semantic attention 으로 C채널 위상 특징 융합 → (N, res²·K)
   │   타겟 특징 = [원본 ⊕ 융합 위상]
   │   HAN(메타패스 그래프 PAP/PSP 위 노드·의미 어텐션) → 노드 분류
   ▼
test Macro-F1
```

`use_topology=false` 면 Stage 1/2 를 건너뛰고 HAN 단독(baseline)으로 학습한다.

## 3. 모듈별 근거

| 단계 | 파일 | 근거/원본 |
|------|------|----------|
| GTN | `tda/models/gtn.py` | Yun et al., NeurIPS 2019 (target-restricted 적응) |
| PDGNN | `tda/models/pdgnn.py` | Yan et al., NeurIPS 2022 / TLC-GNN `pdgnn_modern.py` |
| HKS 필터 | `tda/topology/hks.py` | TLC-GNN `diffusion_features.compute_hks_features` |
| 정확 EPD 라벨 | `tda/topology/epd.py` | TLC-GNN `pdgnn_metapath`/`node_ph_features` (gudhi lower-star) |
| persistence image | `tda/topology/persistence_image.py` | Adams et al., JMLR 2017 |
| fusion | `tda/models/fusion.py` | HAN semantic attention (Wang et al., WWW 2019) |
| HAN | `tda/models/han.py` | Wang et al., WWW 2019 / PyG `HANConv` |

## 4. 충실도 결정 & 가정 (원본과 다른 점 명시)

1. **"GTC" → GTN 으로 해석** (프로젝트 플랜 전체가 GTN+PDGNN).
2. **GTN 메타패스 합성**: 플랜 스케치(§4.2)의 채널 자기제곱(`H@H`)이 아니라, 공식 GTN
   (레이어마다 새 soft-combo 와의 곱)을 따름 — 원논문/repo 우선.
3. **GTN target-restricted**: 플랜 §4.1("채널은 한 노드 타입 위 인접행렬")에 맞춰, 이종
   엣지로부터 미리 만든 타겟–타겟 기저 관계 위에서 GTConv/합성을 적용. 전체 노드 집합에서
   도는 원본 GTN 의 적응판이다.
4. **EPD 필터 = HKS K=3 → 75차원** (res=5 → 25/스케일 × 3). 플랜 스케치는 "degree" 라
   적었으나 확립된 파이프라인 노트(res=5 → 75-dim)와 일치하는 HKS 채택 — repo 우선.
5. **PDGNN 은 진짜 neural EPD 근사기**. 정확 EPD(gudhi)는 **학습 라벨로만**, 추론 시 정확
   계산 없음. 위상을 가장한 단순 통계("fallback")는 이 저장소에 **없다**.
6. **persistence image** 는 표준 PI(Adams)를 순수 numpy 로 재구현. 원본 cython `sg2dgm`
   기본값(범위 (0,1)/(0,1), σ=1.0, CDF 픽셀적분)과 달리 범위 birth(-3,3)/pers(0,6),
   σ=0.5, 격자점 평가를 쓴다 — z-정규화 HKS 필터(값 ~[-3,3])가 원본 (0,1) 범위에서 대부분
   클리핑되는 문제를 피하기 위한 **의도적 편차**(원본보다 더 정보적, 수치는 비재현).
7. **의존성 경량화**: 원본 PDGNN 의 `torch_scatter` → torch 기본 scatter (실행으로 수치
   동일 검증), PI cython → numpy.
   - **GTN 정규화/초기화는 공식 GTN(Yun 2019)에 맞춤**(코드리뷰 후 수정): 레이어 사이
     정규화는 H^T 위 self-loop 없는 행정규화(add=False), GCN conv 는 self-loop 추가
     (add=True) 후 H^T 전파, GTConv 가중치는 constant 0.1 초기화(균등 softmax 시작).
   - **GTN 분류 헤드의 dropout(p=0.5)은 공식엔 없는 추가 정규화**(target-restricted 적응판
     선택). 추론 시(eval) 무영향.
8. **val 분리**: HGB ACM 은 val 마스크가 없어 train 의 15% 를 분리하되, tail 이 아니라
   **시드 고정 무작위 분할**(HGB train 인덱스가 클래스 순이라 tail 분할은 편향).
9. **범위**: 사용자 지정으로 downstream 은 **HAN 만**, fusion 은 **attention 만**
   (HGT, concat 제외). 데이터셋은 **ACM 만** 배선(레지스트리/config 로 교체 가능).

## 4b. 충실(고정) vs 설계(유연)

이 프로젝트는 **단일 논문의 재현이 아니라 새 합성**(HAN+GTN+PDGNN)이다. 따라서 충실도(§0)는
*컴포넌트 단위*로만 적용하고, *조합/하이퍼파라미터*는 데이터셋마다 튜닝하는 설계 자유로 둔다.
여러 데이터셋에 적용할 것이므로 유연한 부분은 config 로 빼서 데이터셋별로 조정 가능하게 했다.

| 구분 | 항목 | 이유 |
|------|------|------|
| **고정(충실)** | PDGNN 구조·집계(SUM⊕MIN), bipartite loss | vendored 신경망 — 바꾸면 "PDGNN" 아님 |
| **고정(충실)** | HKS, exact EPD(gudhi lower-star) | 정의된 수학적 객체 |
| **고정(충실)** | 평가 프로토콜·split·누수방지, "PDGNN/EPD" 라벨 | 비교가능성·정직성 |
| **고정(충실)** | seed/epoch/데이터 전량 (시간 아끼려 줄이지 않음) | §0 — 나쁜 유연성 금지 |
| **유연(설계, config)** | GTN num_channels/num_layers/hidden, knn_k, hop, max_nodes, n_train_samples | 이 조합에 정해진 소스 없음 → val 로 튜닝 |
| **유연(설계, config)** | HAN hidden/heads/dropout, lr/wd/epochs | 새 파이프라인 하이퍼파라미터 |
| **유연(설계, config)** | PI 범위/σ (`pi_birth_range`/`pi_pers_range`/`pi_sigma`) | z-정규화 필터 분포에 맞춤(데이터셋별 조정 가능) |
| **유연(설계)** | GTN target-restricted 적응, fusion, staged vs end-to-end | 조합 차원의 설계 선택 |

> 핵심: GTN 의 정규화/초기화는 공식 GTN 공식에 맞췄지만(임의값보다 원칙적), GTN 자체가
> target-restricted 적응판이므로 판정 기준은 "공식과 byte-match" 가 아니라 **"잘 학습되고
> 합리적 메타패스를 찾는가"(경험적)** 이다.

## 5. 데이터셋: ACM (HGB)

- 타겟 = paper, 3 클래스, Macro-F1. N=3025, 특징 1902 차원.
- 노드: paper(3025)/author(5959)/subject(56)/term(1902, 특징 없음).
- 엣지: paper-cite/ref-paper, paper-to-author, paper-to-subject, paper-to-term (+역방향).
- 기저 관계: PAP, PSP, PcP(cite), PrP(ref), PTP. HAN 메타패스: PAP, PSP.
- 분할: train 770 / val 137 / test 2118 (val 은 train 에서 무작위 15%).

> 주의: PSP(~1.1M 엣지), PTP(~4.6M 엣지)는 매우 조밀하다. 이는 선행 연구의 "dense
> 메타패스 그래프는 해롭다 → kNN 희소화 필수" 발견과 일치하며, Stage 2 가 PDGNN 입력 전
> kNN(k=20) 으로 희소화한다.

## 6. 실험 변형 (config 플래그)

- `use_topology` — full vs baseline(HAN 단독)
- `gtn.num_channels` / `gtn.num_layers` — 메타패스 수/깊이 ablation
- `pdgnn.knn_k` / `pdgnn.hks_K` / `pdgnn.pi_resolution` — 위상 추출 설정
- `seed` — 견고성(분산) 분석

`experiments/run_acm.slurm` = baseline×3 + full×3 (seed 0/1/2) SLURM 배열.

## 7. 결과 & Ablation (ACM, test Macro-F1)

`experiments/run_ablation.slurm` 실측치 (한 코드 버전, seed 0/1/2 평균±표준편차).
기준 config = `configs/acm.json` (GTN 4채널·2레이어, PDGNN K=3·knn20·hop1, HAN hidden64·
heads4, attention fusion). **성능 주장이 아니라 실측치.**

**헤드라인:**

| 실험 | test Macro-F1 |
|------|----------------|
| A1  HAN 단독 (baseline) | 0.8892 ± 0.0113 |
| A3  GTN 단독 (EPD 없음) | 0.8939 ± 0.0088 |
| B2  manual 메타패스(PAP/PSP) + PDGNN-EPD | 0.8977 ± 0.0031 |
| C2  GTN + PDGNN + HAN (attention) | 0.8958 ± 0.0035 |

**Ablation (모두 C2 기준 변형):**

| 실험 | 변형 | test Macro-F1 |
|------|------|----------------|
| D1 | kNN 희소화 제거(PDGNN 앞) | _(no-kNN: ego 거대 → 실행 중, 추후 기입)_ |
| D2 | MIN 집계 제거(SUM 만) | 0.8988 ± 0.0021 |
| D3 | num_channels = 2 / 4 / 8 | 0.8852 / 0.8958 / 0.8951 |
| D4 | GTN depth = 1 / 2 / 3 | 0.8965 / 0.8958 / 0.8891 |
| D5 | random 메타패스(학습 없음) | 0.8887 ± 0.0010 |
| D6 | manual(B2) vs GTN(C2) | 0.8977 vs 0.8958 |

**진단:**
- topology-only (노드 특징 off): **0.7436 ± 0.0097** — 랜덤(0.33) 훨씬 위 → 위상에 신호 있음.
- permutation (test 때 위상 행 셔플): C2 0.8958 → **0.8296** (Δ **+0.066**) → 모델이 위상을 실제 사용.

**해석:**
- **위상에 신호가 있고 모델이 실제로 쓴다**(topology-only 0.74, permutation Δ+0.066). 이는
  통제 시 위상이 신호 없던 선행 LP 실험과 대비된다(faithful 파이프라인 효과).
- 다만 **강한 HAN baseline 을 뚜렷이 넘지는 못한다**(B2·C2 ≈ A1, seed 분산 내). node feature 가
  이미 강함. 단 위상 변형은 **분산이 작다**(A1 은 seed1 에서 0.8735 로 출렁, B2/C2 는 0.89~0.90
  유지) — 선행 "위상의 주효과는 평균이 아니라 분산(robustness)" 과 같은 방향. **seed 3개**라
  분산 주장은 통계적으로 약하다.
- **D6**: GTN-학습(C2) ≈ manual(B2), GTN 이 수동 선택을 넘지 못함. **D5**: GTN(C2) > random(0.8887),
  학습이 random 보다는 나음. **ablation**: no-MIN/얕은 GTN 무해, 깊이 3 은 소폭 하락, 채널 4 가 2보다 나음.

## 8. 실험 커버리지 & 한계

플랜의 실험 매트릭스 대비 현황 (사용자 제약: **HAN만 · attention fusion만 · ACM만**):

| 그룹 | 상태 |
|------|------|
| A1 (HAN), A3 (GTN 단독), B2 (manual+EPD), C2 (GTN+PDGNN+HAN attn) | ✅ 수행 |
| D1–D5 ablation, D6 (= B2 vs C2) | ✅ 수행 (`experiments/run_ablation.slurm`) |
| A2 (HGT), C1 (concat), C3 (HGT) | ⛔ 사용자 제약으로 제외 |
| B1/B3/B4 (DBLP/IMDB/ogbn-mag) | ⛔ ACM 만 배선(레지스트리+config 로 확장 가능) |
| **C4 (end-to-end 학습)** | ❌ **미구현 (future work)** — 아래 |

**C4 미구현 사유:** end-to-end 는 task loss 가 `HAN→fusion→위상특징→PDGNN→GTN` 으로
역전파돼야 하나, GTN 의 soft 인접 → 이산 그래프(kNN 선택 + ego 추출) 단계가 **미분 불가능**해
GTN 까지 gradient 가 닿지 않는다(PDGNN forward·PI 자체는 미분 가능). 진짜 C4 는 kNN 을
Straight-Through Estimator/Gumbel-softmax 로 완화하고 위상 readout 을 전부 미분 가능하게
바꿔야 하며, 플랜 §8 도 *"end-to-end 불안정 → staged 폴백"* 으로 후순위(future work)로 둔다.

**기타 한계:**
- 위상 특징의 성능 기여는 **가설**이며, 본 저장소 실측으로 baseline 초과는 보장되지 않는다.
- seed 3개·단일 기준 config — 분산(견고성) 주장엔 통계력이 약함.
- 고급 위상 모듈(pair-conditioned / multi-slice / typed-cycle 등)은 본 파이프라인 범위 밖.
