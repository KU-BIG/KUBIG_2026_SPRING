# 실험 로그 — MolHIV TDA

이 작업 세션에서 한 모든 것의 시간순 기록. "지금 코드가 어떻게 생겼는지"는
`CURRENT_APPROACH.md` 참고.

## 1. 환경 설정

- `torch==2.5.1+cpu`가 설치돼 있는데 A100 GPU가 있음 → `torch==2.5.1+cu121`로
  재설치.
- `han_pdgnn_fusion/download_molhiv.py`로 `ogbg-molhiv`를 `molhiv_tda/dataset/`에
  다운로드.
- **버그 수정**: `config.py`가 `SMILES_CSV = .../mapping/hiv.csv`였는데, OGB
  다운로드는 `mapping/mol.csv.gz`만 제공. 경로를 `mol.csv.gz`로 수정 (pandas가
  `.gz`를 자동으로 읽어서 다른 코드 변경 불필요).
- `pyyaml`, `rdkit`, `gudhi` 설치 (여러 스크립트가 요구하는데 기본 환경에 없었음).

## 2. Progress bar

공유 학습 루프(`train/train_utils.py:train_one_epoch/run_training`)와
`han_pdgnn_fusion/train_han_pdgnn.py`에 `tqdm` progress bar 추가. epoch별
loss/valid/test/best/stale와 배치별 중첩 bar 표시. 이 함수들을 재사용하는 모든
스크립트에 적용됨.

## 3. PDGNN baseline — 첫 실행

`train/train_pdgnn.py` 기본값(lr=1e-3, dropout=0.5, BCE): **valid 0.7010,
test 0.6804** (100 epoch, patience 10, 단일 seed).

## 4. 하이퍼파라미터 sweep #1 (BCE loss)

그리드: `lr ∈ {1e-3, 1e-4, 1e-5} × dropout ∈ {0.5, 0.4}`, vanilla PDGNN, 신규
`scripts/sweep_pdgnn_hparams.py` 사용.

진행 중 버그 2개 발견/수정:
- 출력 파일명이 config 기본값과 일치할 때 `pdgnn_baseline.json`으로 떨어져서, sweep의
  조회 경로(`pdgnn_lr{lr}_dropout{dropout}.json`)와 충돌 → 항상 lr/dropout으로
  이름 짓게 단순화.
- sweep의 "결과 파일 있으면 스킵" 로직이 `lr=1e-4, dropout=0.4` 칸에서 수동 CLI
  테스트로 남긴 **stale smoke-test 결과**(2 epoch / 256 샘플)를 주워서 재사용 →
  stale 파일 삭제하고 해당 칸 제대로 재실행.

**결과 (BCE, 각 100 epoch, seed=0):**

| LR | Dropout | Valid AUC | Test AUC |
|---:|---:|---:|---:|
| **1e-4** | **0.4** | **0.8184** | **0.7419** |
| 1e-4 | 0.5 | 0.7391 | 0.7086 |
| 1e-3 | 0.4 | 0.7120 | 0.7094 |
| 1e-3 | 0.5 | 0.7010 | 0.6863 |
| 1e-5 | 0.5 | 0.3657 | 0.3003 |
| 1e-5 | 0.4 | 0.3624 | 0.2947 |

`lr=1e-4, dropout=0.4`가 명확한 승자. `lr=1e-5`는 100 epoch 안에 학습 실패
(AUC ≈ 랜덤).

## 5. HAN+PDGNN fusion — 성능 디버깅 (미완료)

`han_pdgnn_fusion/` 탐색 (원자번호 이종 그래프 위 HAN + PDGNN bond-filtration
encoder + cross-attention fusion).

- 첫 1-epoch 시간(전체 데이터셋): **24분 33초**, GPU 사용률 ~13%에 불과.
- 원인 #1: `homo_to_hetero()`(순수 파이썬 노드/엣지 루프)가 그래프 구조가 epoch마다
  안 바뀌는데도 매 그래프·매 배치·매 epoch마다 처음부터 재계산됨.
  - 수정: `hetero_transform.build_hetero_cache` / `load_or_build_hetero_cache`로
    41,127개 그래프의 이종 변환을 한 번만 계산해 `hetero_atomic_number.pt`에 캐싱
    (~354MB, 1회 ~2분).
  - 결과: 소폭 개선(21분 31초/epoch)에 그침 — 캐싱만으로는 부족.
- 원인 #2: `encode_han()`이 미니배치의 그래프 32개를 **하나씩 개별**(파이썬 루프,
  그래프마다 별도 작은 forward)로 처리, 하나의 HAN forward로 배칭 안 함.
  - 수정: `encode_han()`을 재작성해 그래프별 `HeteroData`를 `Batch.from_data_list`로
    묶고, `HANEncoder.encode_hetero`를 미니배치당 **한 번**만 실행, `data.ptr`로
    결과를 homogeneous 노드 순서로 scatter (PyG의 `HeteroData` 배칭이 임의
    `global_index` 속성을 자동 증가시키지 *않음*을 실험으로 확인 → 절대 인덱스를
    `ptr[batch_within_type] + local_global_index`로 직접 계산해야 했음).
- **정확성 버그 발견/수정**: `HANLayer._get_relation_conv`가 관계별 attention
  서브모듈을 **첫 forward 때 lazy 생성**하는데, `train_han_pdgnn.py`의 optimizer는
  forward 이전에 `model.parameters()`로 생성됨 — 즉 관계별 attention 가중치(HAN이
  학습해야 할 실제 "metapath" 메커니즘)가 **optimizer에 안 들어가서 한 번도 업데이트
  안 됨**, 내내 랜덤 초기값에 머묾. `_sync_optimizer_params()`로 수정 (매 배치
  `loss.backward()` 후 `optimizer.step()` 전에 호출, 새로 생긴 파라미터를
  `add_param_group`으로 추가).
- `train_han_pdgnn.py`에도 tqdm progress bar와 `--dropout` CLI 옵션 추가.
- **상태: 배칭 + optimizer 수정 후 재벤치/완전 학습 안 함** — 다른 실험으로 넘어감.
  배칭 수정 *이전* 1-epoch 결과: valid 0.6913, test 0.6028 (24분30초).

## 6. 회전 불변성 논의

(코드 변경 없음) 다음을 명확히 함:
- vanilla PDGNN baseline과 HAN+PDGNN fusion은 **3D 좌표를 전혀 안 씀**(범주형
  원자/결합 feature만) — 회전할 게 없으니 자명하게 불변.
- 별개의, 그때까지 안 쓴 파이프라인(`train_pdgnn_tda.py` + `conformer_3d.py` +
  `pdgnn_3d_dist_mw.py`)은 실제 3D conformer 좌표(RDKit ETKDG)를 계산하고 filtration에
  **pairwise 유클리드 엣지 거리**를 씀 — 이는 본질적으로 회전/이동 불변 (TDA/Rips
  방법이 설계상 의존하는 성질).

## 7. 3D-distance("회전 불변") PDGNN 변형 실행

- `scripts/preprocess_molecular_weight.py` 실행 (43초, RDKit 파싱 실패 7/41,127).
- `scripts/preprocess_3d_edge_dist.py` 실행 (분자별 RDKit ETKDG conformer + MMFF
  최적화, ~26분, 실패 271/41,127 = 0.7% — 대규모 화학 데이터셋에서 정상 범위).
- `train_pdgnn_tda.py`에 `--lr`/`--dropout` CLI 옵션 추가 (이전엔 config 기본값 고정).
- `train_pdgnn_tda.py --config pdgnn_3d_dist_mw`를 튜닝된 lr=1e-4/dropout=0.4로 실행.
- **이 실행은 환경/컨테이너 재시작으로 학습 중 강제 종료됨**, 완료 못 함. stale한
  2-epoch/256-샘플 smoke-test 결과만 `results/pdgnn_3d_dist_mw.json`에 남음
  (valid 0.34/test 0.22 — 무의미, 재실행 필요).

## 8. 세션 재시작

새 날/세션; `/tmp` scratchpad는 지워짐(백그라운드 로그, 진행 중이던 3D-dist 학습)
그러나 프로젝트 디렉토리, GPU torch 설치, 데이터셋, feature 캐시, 모든
`results/*.json`은 정상 유지됨.

## 9. TDA 전략 논의

TDA 노력을 어디에 투자할지 논의:
- Late-fusion(단순 concat) vs. attention 기반 fusion.
- 그래프 위상 filtration(H1 persistence로 고리/cycle 탐지) vs. 3D-distance
  filtration — 그래프 위상은 RDKit conformer 노이즈/실패를 완전히 피함.
- 완전한 SE(3)/O(3)-등변 표현론 레이어(e3nn/Tensor-Field-Network 스타일) vs. 단순
  불변 스칼라 feature도 잠깐 논의 — MolHIV 같은 그래프 분류(불변 출력) 과제에는
  복잡도 대비 가치 없다고 판단.

## 10. Focal loss 실험

- `train_utils.py`에 `focal_loss_with_logits(logits, targets, gamma=2.0,
  alpha=0.25)` 추가; `train_one_epoch`/`run_training`에 optional `loss_fn` 인자를
  넘김 (안 주면 기본 BCE라 다른 호출부는 영향 없음).
- `train_pdgnn.py`에 `--loss {bce,focal}`, `--focal-gamma`, `--focal-alpha` 추가;
  출력 파일명에 `_focal` suffix가 붙어 BCE sweep 결과를 덮어쓰지 않음.
- 같은 6칸 lr×dropout sweep을 focal loss(γ=2.0, α=0.25)로 재실행:

| LR | Dropout | Valid AUC | Test AUC |
|---:|---:|---:|---:|
| **1e-4** | **0.5** | **0.7519** | **0.7236** |
| 1e-3 | 0.5 | 0.7265 | 0.7145 |
| 1e-3 | 0.4 | 0.7253 | 0.6942 |
| 1e-4 | 0.4 | 0.3909 | 0.2263 |
| 1e-5 | 0.4 | 0.3721 | 0.3196 |
| 1e-5 | 0.5 | 0.3691 | 0.3053 |

**핵심 발견**: BCE에서 *최고*였던 칸(lr=1e-4, dropout=0.4, valid 0.8184/test
0.7419)이 focal loss에서 **붕괴**(valid 0.39/test 0.23, 거의 랜덤). focal 자체
최고 칸(1e-4/0.5)도 BCE 최고보다 크게 밀림(valid -0.067, test -0.018).

**결정**: plain BCE 유지. focal loss(이 γ/α 기본값)는 이 데이터/모델에서 정확도
이득 없이 불안정성만 유발 — MolHIV의 ~3.5% 양성 비율은 그리 극단적이지 않고, 평가
지표(랭킹 기반 ROC-AUC)는 detection류처럼 focal의 hard-example 재가중이 필요 없음.

## 11. Baseline 확정

- `config.py`: `LR` 1e-3 → **1e-4**, `DROPOUT` 0.5 → **0.4**.
- `train_pdgnn.py` / `scripts/sweep_pdgnn_hparams.py`: 기본 `--loss`를 `focal`에서
  **`bce`**로 되돌림.
- 기준 수치 확정: **valid 0.8184 / test 0.7419** (단일 seed 0).

## 12. HAN+PDGNN fusion 아키텍처 명확화

(코드 변경 없음) `han_pdgnn_fusion`에서 PDGNN은 이종 그래프를 실제로 안 건드림 —
`PDGNNFiltrationEncoder.forward(x, edge_index, edge_attr)`는 **순수 homogeneous**
OGB 텐서를 받음. HAN 브랜치만 이종 타입 그래프를 소비하고, 두 노드 레벨 스트림
(HAN의 `Z`, PDGNN의 `H`)을 concat(`FusionMLP`) 후 cross-attention으로 결합 —
즉 진짜 late fusion. 이 로그의 다른 곳에서 튜닝 중인 plain-PDGNN baseline과는
구조적으로 다른 별개 모델.

## 13. 실제 persistent-homology(고리 인지) bond-TDA feature

사용자가 HAN late-fusion 경로를 건너뛰고 대신 **vanilla PDGNN의 filtration
feature**를 고리/cycle 위상을 제대로 잡도록 업그레이드하자고 요청.

- `features/bond_filtration_tda.py`가 **이미 진짜 gudhi persistent homology**(H0 +
  H1, `gudhi.SimplexTree`, bond-type filtration)를 구현 중임을 발견 — H1로 고리
  구조를 잡음. 이는 `han_pdgnn_fusion/filtration.py`의 threshold degree/sum/mean
  노드 통계(진짜 persistent homology 아님)와 다르고 더 낫다. (파일을 실제로 읽기
  전 대화 중간에 잘못 말한 것을 정정함.)
- 이건 `models/pdgnn_tda.py::PDGNNTDA`로 들어가고, classifier head에서 단순
  concat(attention 아님) — 요청한 방식과 일치.
- `scripts/preprocess_bond_tda.py` 실행 (41,127개 28초, 50차원 출력).
- `train_pdgnn_tda.py --config pdgnn_bond_tda --lr 1e-4 --dropout 0.4` 실행 (BCE,
  100 epoch): **valid 0.7404, test 0.7370**.

당시엔 baseline(0.8184/0.7419)을 못 이겼다고 봤으나, **나중에 이 결과가 무효임이
밝혀짐** — persistence-image 버그(§16)로 feature가 사실상 0이었음.

## 14. Multi-seed baseline vs BondTDA (seed 0,1,2)

단일 seed 노이즈가 결론을 흔든다는 우려에서, baseline과 BondTDA를 3 seed로 재실행.
`train_pdgnn.py`/`train_pdgnn_tda.py`에 `--seed` 옵션과 seed suffix 파일명 추가,
`scripts/multiseed_baseline_vs_bondtda.py` 작성.

| 모델 | Valid AUC | Test AUC |
|---|---|---|
| PDGNN baseline | 0.7714 ± 0.0354 | 0.7296 ± 0.0110 |
| PDGNN + BondTDA | 0.7563 ± 0.0146 | 0.7345 ± 0.0021 |

seed 0의 baseline 0.8184는 낙관적 outlier(seed별 valid 0.7330~0.8184)로 확인.
3 seed로 보면 BondTDA가 test 평균 약간 높고 분산이 ~5배 낮았음. **단, 이 비교는
버그난 persistence image로 돌린 것이라 이후 무효 판정**(§16).

실행 중 컨테이너 재시작으로 baseline seed=2가 한 번 중단됐다가, "결과 있으면 스킵"
로직 덕분에 이어서 재개함.

## 15. Epoch 기본값 축소

머신이 느려져 학습이 오래 걸려서, `config.py`의 `EPOCHS` 100 → **50**, 그리고
sweep/multiseed/ablation 스크립트의 기본 `--epochs`도 50으로 낮춤. (현재 실험은
30 epoch으로 스크리닝.)

## 16. Persistence-image 버그 발견/수정 (중요)

그래프 위상 filtration을 만들다가(§17), 방향족 고리가 분명한 분자인데 persistence
image가 **전부 0**으로 나오는 걸 발견. persistence diagram 자체는 정상(H1 loop가
birth=1, death=2로 잡힘)이었으나, 벡터화가 문제였음:

- `PI_SIGMA = 0.05`가 여기서 쓰는 정수 스케일 filtration 값에 비해 너무 좁아서,
  persistence point가 grid 셀 사이에 떨어져 exp(-거리²/2σ²) ≈ 0이 됨.
- 확인 결과 **기존 `bond_tda.pt` 캐시도 ~98%가 0**이었음 → 즉 §13, §14의
  "PDGNN + BondTDA" 실험은 사실상 **죽은(near-zero) concat 브랜치를 단 vanilla
  PDGNN**이었고, BondTDA가 baseline과 비슷했던 게 이 때문.
- **`PI_SIGMA = 0.5`(픽셀 절반)로 수정**하고 모든 TDA 캐시 재생성. H1이 고리 함량을
  잘 추적함을 검증: 융합 방향족 분자(graph 2) H1 높음, 고리 1개(graph 5) 중간,
  비고리 사슬(graph 10) H1=0.

## 17. Multi-filtration ablation 설계 및 스크리닝

사용자와 논의해 "그래프 위상 구조를 최대한 살리는" 방향으로, filtration 렌즈 3종을
만들고 각 임베딩을 concat하는 ablation 설계:

- **A** = 전체 그래프 hop-distance Rips (`features/graph_rips_tda.py`, 신규) — 고리
  크기 위상.
- **B** = 방향족 결합만 남긴 hop-distance Rips (신규) — 방향족 고리 시스템.
- **C** = 기존 bond-type filtration (`bond_tda.pt`) — 결합 차수 위상.

각 렌즈는 50차원 H0+H1 persistence image. `train/train_pdgnn_multifiltration.py`가
선택된 렌즈를 concat해 `PDGNNTDA` head에 넣음. 도메인 관점에서 이 데이터셋의 핵심은
scaffold split(낯선 고리 골격으로의 일반화)이고, 고리 시스템이 활성의 핵심 신호라
이 접근이 데이터 본질에 맞음.

8칸 스크리닝(vanilla, A, B, C, A+B, B+C, C+A, A+B+C; seed 0, 30 epoch), test 정렬:

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

당시 관찰: A+B 빼고 모두 vanilla를 test에서 이김(A+B+C가 +0.031). C(bond-type)가
test 일반화의 핵심으로 보임. valid와 test가 상위권에서 반대로 감(A는 valid 최고지만
test 나쁨 — scaffold-split 과적합).

## 18. Multi-seed 확정 (상위 3개 + vanilla)

스크리닝 상위 조합 간 차이가 노이즈(±0.011)보다 작아, `confirm_multifiltration.py`로
A+B+C, B+C, C, vanilla를 seed 0,1,2로 확정:

| Filtrations | Valid (mean±std) | Test (mean±std) | seed별 test |
|---|---|---|---|
| A+B+C | 0.7610 ± 0.0029 | 0.7301 ± 0.0057 | 0.7373, 0.7234, 0.7296 |
| B+C | 0.7565 ± 0.0064 | 0.7300 ± 0.0087 | 0.7345, 0.7178, 0.7375 |
| vanilla | 0.7474 ± 0.0107 | 0.7273 ± 0.0145 | 0.7067, 0.7368, 0.7382 |
| C | 0.7280 ± 0.0138 | 0.7265 ± 0.0047 | 0.7315, 0.7202, 0.7278 |

**정직한 결론 — 스크리닝의 "TDA 승리"는 대부분 seed-0 아티팩트**:
- Test AUC로는 TDA가 vanilla를 유의미하게 못 이김(A+B+C +0.003, vanilla ±0.0145
  범위 안). 스크리닝의 "+0.031"은 seed-0 vanilla가 운 나쁘게 낮았던 탓.
- **진짜 견고한 효과는 분산 감소**: test std vanilla ±0.0145 vs A+B+C ±0.0057 /
  C ±0.0047 — ~2.5–3배 감소. valid도 동일.
- valid는 TDA가 일관되게 높으나 test로 안 이어짐 → 가벼운 과적합.

방어 가능한 주장: **"위상 feature는 regularizer로 작동 — scaffold-split 일반화의
안정성(분산 감소)을 주지, 평균을 올리진 않는다."**

## 19. A+B+C dropout sweep

TDA 브랜치 과적합을 dropout으로 잡을 수 있는지 확인. `train_pdgnn_multifiltration.py`
파일명에 dropout suffix(`_d0.5` 등) 추가, `scripts/sweep_dropout_multifilt.py` 작성.
dropout {0.5, 0.6, 0.7} × seed {0,1,2} (0.4는 §18 재사용):

| Dropout | n | Valid (mean±std) | Test (mean±std) | seed별 test |
|---:|---:|---|---|---|
| **0.4** | 3 | **0.7610 ± 0.0029** | **0.7301 ± 0.0057** | 0.7373, 0.7234, 0.7296 |
| 0.5 | 3 | 0.7486 ± 0.0025 | 0.7114 ± 0.0082 | 0.7143, 0.7002, 0.7197 |
| 0.6 | 3 | 0.7444 ± 0.0046 | 0.7082 ± 0.0075 | 0.7162, 0.6982, 0.7101 |
| 0.7 | 2 | 0.7208 ± 0.0136 | 0.7009 ± 0.0085 | 0.7094, 0.6924 |

**결론: dropout을 세게 걸수록 valid·test 단조 감소.** 가설(강한 dropout이 과적합을
잡아 test를 올림)은 틀림 — 오히려 언더피팅. **dropout 0.4가 이미 최적.** (사용자
요청으로 0.7은 seed2 생략, 2 seed만 실행.)

## 20. 렌즈 D — Gasteiger 부분 전하 filtration ("정전기 정보" 싼 버전)

팀 회의에서 "정전기적 인력 같은 외부 정보"를 넣자는 제안. 이걸 두 갈래로 정리:
(a) 부분 전하 자체 = 2D, 쌈 / (b) 진짜 through-space Coulomb qᵢqⱼ/rᵢⱼ = 3D 필요.
먼저 싼 (a)로 "3D 없이 얼마나 짜낼 수 있나"의 상한을 측정.

- **신규 렌즈 D**(`features/charge_filtration_tda.py`): RDKit Gasteiger 전하를 노드
  스칼라로 놓고 **sublevel-set(lower-star) filtration** — 노드는 자기 전하값에서,
  엣지는 양 끝 max에서 등장. H0가 전기음성(저전하) basin 위상, H1이 고리를 잡음.
  회전/이동 불변(2D 그래프의 결정론적 함수).
- **스케일 함정 회피**: 전하는 ~[-0.6,0.6] 스케일이라 정수용 `PI_SIGMA=0.5`를 쓰면
  persistence image가 뭉개짐(§16 all-zero 버그의 역방향). 전하 전용
  `CHARGE_PI_SIGMA=0.12`, `CHARGE_BIRTH_RANGE=(-0.6,0.6)`,
  `CHARGE_PERS_RANGE=(0,1.2)` 신설. 전처리에 nonzero-fraction sanity check 내장.
- `scripts/preprocess_charge_tda.py`로 41,127개 캐시(`charge_tda.pt`), ~37초.
  검증: nonzero_frac 0.487, max 62 — 건강. SMILES 실패 7/41,127(정상).
- `train_pdgnn_multifiltration.py`에 렌즈 D 등록. D / C+D / A+B+C+D를 seed 0,1,2,
  30 epoch으로 실행:

| Filtrations | Test (mean±std) | seed별 test | 기존 비교 |
|---|---|---|---|
| C+D | 0.7315 ± 0.0078 | 0.7376, 0.7363, 0.7206 | A+B+C 0.7301±0.0057 |
| **A+B+C** | 0.7301 ± 0.0057 | (§18) | — |
| vanilla | 0.7273 ± 0.0145 | (§18) | — |
| **D** | **0.7267 ± 0.0017** | 0.7256, 0.7291, 0.7253 | vanilla 0.7273 |
| C | 0.7265 ± 0.0047 | (§18) | — |
| A+B+C+D | 0.7168 ± 0.0031 | 0.7150, 0.7211, 0.7142 | A+B+C 0.7301 |

**결론 — 렌즈 D는 ROC-AUC를 못 올림, 예측대로:**
- **D 단독: 평균이 정확히 vanilla 수준(0.7267 vs 0.7273), 정확도 이득 0.** 그러나
  **분산이 지금까지 모든 config 중 최저(±0.0017)** — vanilla ±0.0145의 ~8배 감소,
  A+B+C ±0.0057보다도 낮음. Gasteiger 전하가 2D 그래프의 결정론적 함수라 "새 정보가
  아니라 explicit inductive bias"라던 사전 예측과 정확히 일치 → **가장 순수한
  regularizer 예시.**
- **C+D: 0.7315 ± 0.0078.** 첫 2 seed(0.7376, 0.7363)만 보면 최고 조합처럼 보였으나
  seed2가 0.7206으로 떨어져 평균이 A+B+C와 사실상 동률(+0.0014, 노이즈 안). 또
  seed-luck 아티팩트. 분산도 A+B+C보다 큼.
- **A+B+C+D: 0.7168 ± 0.0031 — 풀스택에 D를 얹으면 평균이 ~0.013 하락**(A+B+C
  0.7301 대비, 두 std 밖). concat head에 중복 차원이 늘며 dilution/과적합.

**종합: 전하 렌즈는 정확도를 안 올리고(오히려 풀스택에선 낮춤), "위상 feature =
안정성 prior(분산 감소)" 스토리만 강화.** through-space 정전기(b)가 아니면 3D 없는
전하 정보로 AUC를 올릴 길은 없다는 것을 실측으로 확인. AUC 상승은 다른 축(fingerprint
concat, backbone, 앙상블)에서 찾아야 함.

## 미해결 / 안 한 것

- HAN+PDGNN fusion: 배칭 + optimizer 수정 후 end-to-end 재벤치 안 함.
- 3D-distance PDGNN 변형(`pdgnn_3d_dist_mw`): 재시작으로 중단, 튜닝값으로 완료 못 함.
- TDA가 test 평균을 올리는지 vs 분산만 줄이는지: 현재 근거로는 분산 감소만.
  seed를 더(n=5) 늘리거나 다른 축(lr, 벡터화 해상도)을 봐야 확정 가능.
- filtration 설계 개선(예: 화학적 결합 길이 가중 hop-distance Rips)이 분산 감소를
  강화하는지 미탐색.
