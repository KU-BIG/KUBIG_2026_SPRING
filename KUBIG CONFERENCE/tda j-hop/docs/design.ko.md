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
6. **persistence image** 는 표준 PI(Adams)를 순수 numpy 로 재구현 — 원본 cython `sg2dgm`
   의 수치를 그대로 재현한 것은 아님(공유/Colab 호환).
7. **의존성 경량화**: 원본 PDGNN 의 `torch_scatter` → torch 기본 scatter (수치 동일),
   PI cython → numpy.
8. **val 분리**: HGB ACM 은 val 마스크가 없어 train 의 15% 를 분리하되, tail 이 아니라
   **시드 고정 무작위 분할**(HGB train 인덱스가 클래스 순이라 tail 분할은 편향).
9. **범위**: 사용자 지정으로 downstream 은 **HAN 만**, fusion 은 **attention 만**
   (HGT, concat 제외). 데이터셋은 **ACM 만** 배선(레지스트리/config 로 교체 가능).

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

## 7. 결과 (ACM, test Macro-F1)

> 실행 후 `runs/acm_*/metrics.json` 에서 채운다. 성능 주장이 아니라 **실측치**이며, 선행
> 연구의 "정확한 위상 ≠ 유용한 위상" 및 "위상의 주효과는 분산 감소" 맥락에서 해석한다.

| 설정 | seed0 | seed1 | seed2 | 평균 ± 표준편차 |
|------|-------|-------|-------|----------------|
| baseline (HAN) | … | … | … | … |
| full (GTN+PDGNN+HAN, attn) | … | … | … | … |

## 8. 한계

- staged 학습(Option A)만 구현 — end-to-end(플랜 Option B, 미분가능 EPD 필요)는 미구현.
- GTN 의 발견 메타패스가 수동 선택보다 낫다는 직접 비교(B vs C)는 아직 수행 안 함.
- 위상 특징의 성능 기여는 **가설**이며, 본 저장소 실측으로 baseline 초과는 보장되지 않는다.
- 고급 위상 모듈(pair-conditioned / multi-slice / typed-cycle 등)은 본 파이프라인 범위 밖.
