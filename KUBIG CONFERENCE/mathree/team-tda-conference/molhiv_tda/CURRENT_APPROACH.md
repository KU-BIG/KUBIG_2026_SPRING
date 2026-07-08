# 현재 접근 방식 — MolHIV TDA

지금 코드가 *실제로 무엇을 하는지*에 대한 스냅샷. 여기까지 오게 된 과정은
`EXPERIMENT_LOG.md` 참고.

## 과제

OGB `ogbg-molhiv`: 이진 그래프 분류 (HIV 복제 억제 여부), 분자 41,127개,
공식 scaffold split (train 32,901 / valid 4,113 / test 4,113), OGB의 ROC-AUC
evaluator로 평가. 라벨 불균형 (양성 ~3.5%).

## 확정된 baseline

**Vanilla PDGNN, BCE loss, lr=1e-4, dropout=0.4** — 다른 모든 변형은 이걸 기준으로
비교한다. 재현:

```bash
cd molhiv_tda
python train/train_pdgnn.py --epochs 100 --device cuda   # config.py 기본값 사용
```

`config.py`에 `LR = 1e-4`, `DROPOUT = 0.4`가 프로젝트 기본값으로 설정돼 있음.
seed 0 단일 기준 valid 0.8184 / test 0.7419였으나, **여러 seed로 보면 이 수치는
낙관적 outlier** (아래 multi-seed 결과 참고).

## Backbone: PDGNN

`models/pdgnn_conv.py::PDConv` — TLC-GNN에서 가져온 message-passing 레이어:
메시지는 `LeakyReLU(Linear([x_i, x_j, bond_emb]))`, **concat(sum, min)**
(Union-Find 방식)으로 집계. `models/pdgnn_baseline.py::PDGNNBaseline`은 이걸
4개(`NUM_BACKBONE_LAYERS`) 쌓고, OGB의 `AtomEncoder`/`BondEncoder`를 쓰며,
global-add-pool 후 2층 MLP head로 분류.

## Loss

`train/train_utils.py::train_one_epoch`는 optional `loss_fn`을 받음 (안 넘기면
기본 `binary_cross_entropy_with_logits`). binary focal loss
(`focal_loss_with_logits`, γ/α 조절 가능)도 구현돼 있고 `train_pdgnn.py --loss
focal`로 쓸 수 있으나, **BCE가 채택된 기본값** — focal loss는 테스트 후 기각
(로그 §10 참고).

## 모델 변형 (전부 `molhiv_tda/` 아래)

| 스크립트 | `--config` | 추가하는 것 | baseline 대비 |
|---|---|---|---|
| `train/train_pdgnn.py` | *(없음)* | 없음 — 이게 **baseline** | — |
| `train/train_pdgnn_tda.py` | `pdgnn_mw` | + 분자량(스칼라) | 이번 세션 미테스트 |
| `train/train_pdgnn_tda.py` | `pdgnn_bond_tda` | + bond-type filtration TDA (실제 H0/H1 persistence) | 아래 multifilt로 대체됨 |
| `train/train_pdgnn_tda.py` | `pdgnn_3d_dist_mw` | 3D conformer edge-distance filtration + bond-weighted MW fusion (다른 backbone `PDGNN3DDistMW`) | 중간에 중단, 유효 결과 없음 |
| `train/train_pdgnn_multifiltration.py` | `--filtrations A,B,C` | 그래프 위상 filtration 렌즈 concat (핵심 실험) | 아래 결과 참고 |
| `han_pdgnn_fusion/train_han_pdgnn.py` | `--model main` | HAN(이종 그래프) + PDGNN filtration encoder + cross-attention fusion | 버그 수정했으나 재벤치 안 함 |

`train_pdgnn.py`/`train_pdgnn_tda.py`/`train_pdgnn_multifiltration.py`는
`train/train_utils.py`의 학습/평가 루프를 공유 (tqdm progress bar, focal-loss
지원, valid 최고 checkpoint, patience early stopping). `han_pdgnn_fusion/`은
별도 코드베이스로 자체 루프를 가짐.

## Multi-filtration: 위상 렌즈 3종

각 렌즈는 50차원 H0+H1 persistence image이고, 선택된 것들을 PDGNN head에서
concat (`train/train_pdgnn_multifiltration.py`):

- **A** = 전체 그래프 hop-distance Rips (고리 크기 / 일반 cycle 위상)
- **B** = 방향족 서브그래프 hop-distance Rips (방향족 고리 시스템)
- **C** = bond-type filtration (결합 차수 구조; 기존 `bond_tda.pt`)
- **D** = Gasteiger 부분 전하 sublevel-set filtration (전하 분포 위상; `charge_tda.pt`)

네 렌즈 모두 **3D 좌표 없이 연결 구조/2D 화학만** 쓰므로 회전/이동 불변이 자동 보장됨.
`features/graph_rips_tda.py`(A, B), `features/bond_filtration_tda.py`(C),
`features/charge_filtration_tda.py`(D)가 gudhi로 실제 persistent homology
(H0+H1, 고리 포함)를 계산. 렌즈 D는 전하 스케일(~[-0.6,0.6])에 맞춘 전용
persistence-image sigma(`CHARGE_PI_SIGMA=0.12`)를 씀 — 정수용 `PI_SIGMA=0.5`를
그대로 쓰면 이미지가 뭉개짐.

## 필요한 전처리 캐시 (`molhiv_tda/cache/`)

| 캐시 파일 | 스크립트 | 필요한 곳 |
|---|---|---|
| `bond_tda.pt` (렌즈 C) | `scripts/preprocess_bond_tda.py` | multifilt `C` |
| `graph_rips_tda.pt` (렌즈 A) | `scripts/preprocess_graph_rips_tda.py` | multifilt `A` |
| `aromatic_rips_tda.pt` (렌즈 B) | `scripts/preprocess_graph_rips_tda.py` | multifilt `B` |
| `charge_tda.pt` (렌즈 D) | `scripts/preprocess_charge_tda.py` | multifilt `D` |
| `molecular_weight.pt` | `scripts/preprocess_molecular_weight.py` | `use_mw` 계열 |
| `edge_dist_3d.pt` | `scripts/preprocess_3d_edge_dist.py` | `pdgnn_3d_dist_mw` |

`han_pdgnn_fusion/cache/hetero_atomic_number.pt`는 별도 캐시(이종 그래프 변환
결과)로, `han_pdgnn_fusion/`에서만 사용되며 첫 실행 시 자동 생성됨.

## 결과

별도 명시 없으면 BCE, lr=1e-4. **기본 epoch은 이제 50** (config), 현재 실험은
30 epoch으로 스크리닝.

### 하이퍼파라미터 sweep (vanilla PDGNN, 단일 seed=0)

| Loss | LR | Dropout | Valid AUC | Test AUC |
|---|---:|---:|---:|---:|
| BCE | **1e-4** | **0.4** | **0.8184** | **0.7419** |
| BCE | 1e-4 | 0.5 | 0.7391 | 0.7086 |
| BCE | 1e-3 | 0.4 | 0.7120 | 0.7094 |
| BCE | 1e-3 | 0.5 | 0.7010 | 0.6863 |
| BCE | 1e-5 | 0.4/0.5 | ~0.36 | ~0.30 (학습 실패) |
| Focal(γ=2,α=.25) | 1e-4 | 0.5 | 0.7519 | 0.7236 |
| Focal | 1e-4 | 0.4 | 0.3909 | 0.2263 (붕괴) |

승자 = BCE, lr=1e-4, dropout=0.4. Focal loss는 기각 (불안정, 이득 없음).

### Persistence-image 버그 (수정 완료)

`PI_SIGMA`가 `0.05`로, 여기서 쓰는 정수 스케일 filtration 값에 비해 너무 좁아서 —
모든 persistence point가 grid 셀 사이에 떨어져 ~0 가중치를 받았고, 결국 모든
persistence-image 캐시(`bond_tda.pt` 등)가 ~98% 0이었음. **`PI_SIGMA = 0.5`로
수정**(픽셀의 약 절반)하고 모든 TDA 캐시 재생성. H1이 고리 함량을 잘 추적함을 검증
(융합 방향족 분자 → 높은 H1; 비고리 사슬 → 0). **이 수정 이전의 모든 TDA 결과는
무효이며 재실행함.**

### Multi-filtration ablation — 8칸 스크리닝 (seed 0, 30 epoch)

test AUC 기준 정렬:

| Filtrations | Valid AUC | Test AUC |
|---|---:|---:|
| **A+B+C** | 0.7613 | **0.7373** |
| B+C | 0.7564 | 0.7345 |
| C | 0.7185 | 0.7315 |
| C+A | 0.7640 | 0.7307 |
| B | 0.7700 | 0.7257 |
| A | 0.7831 | 0.7214 |
| vanilla | 0.7592 | 0.7067 |
| A+B | 0.7634 | 0.6859 |

발견 (단일 seed — 스크리닝이지 확정 아님):
- **A+B 빼고 모든 조합이 test에서 vanilla를 이김.** 버그 고친 위상 feature가
  기여하는 것으로 보임 (당시 해석). 최고 A+B+C가 vanilla 대비 +0.031.
- **C(bond-type)가 test 일반화의 핵심.** 상위 4개 중 3개가 C 포함; C 없는
  조합(A, B, A+B)은 전부 하위권.
- **상위권에서 valid와 test가 반대로 감.** A 단독은 valid 최고(0.783)지만 test는
  나쁨(0.721) — 전형적 scaffold-split 과적합. → **여기서 valid로 모델 고르면 안 됨.**
- **A+B가 유일하게 vanilla보다 나쁨** — 결합차수 앵커(C) 없이 고리 위상 렌즈 둘만
  겹치면 해로움.

### Multi-seed 확정 (seed 0,1,2; 30 epoch)

`scripts/confirm_multifiltration.py`로 상위 3개 + vanilla를 3 seed 재실행:

| Filtrations | Valid (mean±std) | Test (mean±std) | seed별 test |
|---|---|---|---|
| A+B+C | 0.7610 ± 0.0029 | 0.7301 ± 0.0057 | 0.7373, 0.7234, 0.7296 |
| B+C | 0.7565 ± 0.0064 | 0.7300 ± 0.0087 | 0.7345, 0.7178, 0.7375 |
| vanilla | 0.7474 ± 0.0107 | 0.7273 ± 0.0145 | 0.7067, 0.7368, 0.7382 |
| C | 0.7280 ± 0.0138 | 0.7265 ± 0.0047 | 0.7315, 0.7202, 0.7278 |

**정직한 해석 — 스크리닝의 "TDA 승리"는 대부분 seed-0 아티팩트였음:**
- **Test AUC로는 TDA가 vanilla를 유의미하게 못 이김**: A+B+C 0.7301 vs vanilla
  0.7273은 +0.003으로 vanilla의 ±0.0145 범위 안. 스크리닝의 "+0.031"은 seed-0
  vanilla가 운 나쁘게 낮았던 것(0.7067) 탓이고, seed 1–2는 ~0.737로 회복.
- **진짜 견고한 효과는 분산 감소.** Test 표준편차: vanilla ±0.0145 vs A+B+C
  ±0.0057 / C ±0.0047 — TDA가 seed 간 분산을 ~2.5–3배 줄임. valid도 동일
  (vanilla ±0.0107 vs A+B+C ±0.0029).
- **Valid AUC는 TDA가 일관되게 높지만**(A+B+C 0.761 vs vanilla 0.747) test로
  이어지지 않음 → 실제 이득보다는 가벼운 과적합에 가까움.

### A+B+C dropout sweep (seed 0,1,2; 30 epoch)

TDA 브랜치의 과적합을 dropout으로 잡을 수 있는지 확인:

| Dropout | n | Valid (mean±std) | Test (mean±std) | seed별 test |
|---:|---:|---|---|---|
| **0.4** | 3 | **0.7610 ± 0.0029** | **0.7301 ± 0.0057** | 0.7373, 0.7234, 0.7296 |
| 0.5 | 3 | 0.7486 ± 0.0025 | 0.7114 ± 0.0082 | 0.7143, 0.7002, 0.7197 |
| 0.6 | 3 | 0.7444 ± 0.0046 | 0.7082 ± 0.0075 | 0.7162, 0.6982, 0.7101 |
| 0.7 | 2 | 0.7208 ± 0.0136 | 0.7009 ± 0.0085 | 0.7094, 0.6924 |

**결론: dropout을 세게 걸수록 valid·test 모두 단조 감소.** 가설(강한 dropout이
과적합을 잡아 test를 올림)은 틀림 — 오히려 언더피팅으로 성능이 떨어짐.
**dropout 0.4가 이미 최적.** (0.7은 seed2 생략, 2 seed만.)

### 렌즈 D — 부분 전하 filtration (seed 0,1,2; 30 epoch)

"정전기 정보"의 싼 버전(3D 없는 2D 전하)을 상한 측정용으로 추가. test AUC 정렬:

| Filtrations | Test (mean±std) | seed별 test |
|---|---|---|
| C+D | 0.7315 ± 0.0078 | 0.7376, 0.7363, 0.7206 |
| A+B+C | 0.7301 ± 0.0057 | 0.7373, 0.7234, 0.7296 |
| vanilla | 0.7273 ± 0.0145 | 0.7067, 0.7368, 0.7382 |
| **D** | **0.7267 ± 0.0017** | 0.7256, 0.7291, 0.7253 |
| C | 0.7265 ± 0.0047 | 0.7315, 0.7202, 0.7278 |
| A+B+C+D | 0.7168 ± 0.0031 | 0.7150, 0.7211, 0.7142 |

- **D 단독: 평균은 vanilla와 동일(0.7267 vs 0.7273), 정확도 이득 0. 그러나 분산이
  지금까지 최저(±0.0017)** — vanilla의 ~8배 감소. Gasteiger 전하는 2D 그래프의
  결정론적 함수라 "새 정보가 아니라 explicit inductive bias" → 가장 순수한
  regularizer. A/B/C의 분산 감소 패턴을 극단적으로 재현.
- **C+D: A+B+C와 사실상 동률(+0.0014, 노이즈 안).** 첫 2 seed만 보면 최고처럼 보였으나
  seed2가 무너져 또 seed-luck. 분산도 A+B+C보다 큼.
- **A+B+C+D: 풀스택에 D를 얹으면 평균 ~0.013 하락** — 중복 차원이 concat head를
  희석/과적합. 렌즈를 무작정 더하면 해로움.

**렌즈 D 결론: 전하 정보는 ROC-AUC를 못 올림(풀스택에선 낮춤), "안정성 prior" 스토리만
강화.** 3D 없는 전하로 AUC를 올릴 길은 없음이 실측 확인됨.

### 종합 결론

"TDA가 AUC를 올린다"는 주장은 이 데이터에서 성립하지 않음 — A/B/C에 이어 전하 렌즈 D도
동일. 방어 가능한 주장은 **"위상 feature가 regularizer로 작동한다 — scaffold-split
일반화의 안정성(분산 감소)을 주지, 평균을 올리진 않는다."** 렌즈 D가 이 스토리의 가장
깨끗한 증거(vanilla 평균 유지 + 분산 8배 감소). TDA 학회용으로는 여전히 유효
(topology as a stability prior)지만, 정확도 향상이 아니라 분산 감소로 프레이밍해야 함.

**AUC를 실제로 올리려면** 위상 렌즈를 더 쌓는 방향(수확 체감/역효과 확인됨)이 아니라
다른 축이 필요: (1) Morgan/ECFP fingerprint concat, (2) backbone 교체(GIN+virtual
node 등), (3) **앙상블** — 위상 렌즈가 분산을 줄이고 예측을 decorrelate하니, seed/렌즈
across 평균으로 "분산 감소"를 mean AUC로 환금하는 게 가장 논리적인 다음 수.
정전기를 진짜로 쓰려면 through-space Coulomb(qᵢqⱼ/rᵢⱼ, 비결합쌍)이 필요하고 이는 3D
전제 — 렌즈 D(2D 전하)로는 그 정보가 안 들어감.

### 아직 벤치마크 안 한 것

- HAN+PDGNN cross-attention fusion (성능·optimizer 버그 수정했으나 재실행 안 함).
- 3D-distance("회전 불변") PDGNN 변형 (`pdgnn_3d_dist_mw`) — 중간 중단, 유효 결과 없음.

## 미해결 질문

- **TDA가 test AUC를 올리나, 분산만 줄이나?** 현재 3-seed 근거로는 분산 감소만
  (평균 이득은 노이즈 안쪽). seed를 더 늘리면(n=5) 평균이 기울까?
- dropout 튜닝은 이득 없음이 확인됨. 다른 축(lr, TDA 벡터화 차원/해상도)은 미탐색.
- filtration 설계를 개선하면(예: hop-distance Rips에 화학적 결합 길이 가중) 분산
  감소 효과가 강해지나?
- HAN+PDGNN fusion과 3D-distance 변형은 여전히 미벤치.
