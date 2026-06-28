# TDA: HAN + GTN + PDGNN 파이프라인

이종(heterogeneous) 그래프에서 **위상 특징(EPD)** 을 자동으로 추출해 노드 분류에 쓰는
하나의 파이프라인입니다. 핵심 질문:

> **위상 특징(persistent homology)이 이종 그래프 학습을 돕는가? 그리고 그걸 가능케 하는
> 메타패스 선택을 자동화할 수 있는가?**

기존 연구에서는 메타패스를 사람이 직접 골랐습니다(예: DBLP 의 APA, APCPA). 이 프로젝트는
**GTN 이 메타패스를 자동 발견**하고, 그 위에서 **PDGNN 이 위상 특징을 계산**하도록 묶습니다.

```
이종 그래프
   │  엣지 타입별 인접행렬
   ▼
[Stage 1] GTN  ── 메타패스 채널 C개 자동 발견 (학습된 인접행렬 H)
   │  채널별 동종(타겟–타겟) 그래프
   ▼
[Stage 2] PDGNN ── 채널마다 EPD 근사 → per-node persistence image (75차원)
   │  C개의 위상 특징
   ▼
[Stage 3] Semantic Attention Fusion → 타겟 특징 강화 → HAN 노드 분류
```

---

## 무엇이 들어있나

| 모듈 | 역할 | 근거 |
|------|------|------|
| `tda/models/gtn.py` | GTN — 메타패스 자동 발견 | Yun et al., NeurIPS 2019 |
| `tda/models/pdgnn.py` | PDGNN — neural EPD 근사기 | Yan et al., NeurIPS 2022 (TLC-GNN repo) |
| `tda/models/han.py` | HAN — downstream 노드 분류 | Wang et al., WWW 2019 (PyG `HANConv`) |
| `tda/models/fusion.py` | 채널 위상 특징 의미 어텐션 융합 | HAN semantic attention |
| `tda/topology/hks.py` | HKS 노드 필터 | TLC-GNN `diffusion_features` 재구현 |
| `tda/topology/persistence_image.py` | persistence diagram → 벡터 | Adams et al., JMLR 2017 |
| `tda/topology/epd.py` | 채널 그래프 → PDGNN 위상 특징 추출 | TLC-GNN `pdgnn_metapath` 재구성 |
| `tda/data/` | 데이터셋 로더 + 기저 관계/메타패스 구성 | HGB benchmark |
| `tda/train.py` | staged 학습 드라이버 | 프로젝트 플랜 Option A |

---

## 설치

```bash
conda create -n tda python=3.9 -y && conda activate tda
pip install -r requirements.txt   # torch/torch_geometric 는 플랫폼 휠 권장
pip install -e .
```

검증 환경: python 3.9, torch 2.1.0+cu118, torch_geometric 2.5.3, gudhi 3.11.
원본 PDGNN 의 `torch_scatter`, persistence image 의 cython(`sg2dgm`) 의존성은
**제거**했습니다(각각 torch 기본 scatter, 순수 numpy 로 대체 — 수치 동일). 별도 빌드 불필요.

## 실행

```bash
# 전체 파이프라인 (GTN + PDGNN + HAN, attention fusion)
python -m tda.train --config configs/acm.json --dataset acm \
    --data-root ./data --output-dir runs/acm_full --seed 0

# baseline (HAN 단독, 위상 없음) — 비교용
python -m tda.train --config configs/acm.json --dataset acm \
    --data-root ./data --output-dir runs/acm_base --seed 0 --no-topology
```

데이터가 `--data-root` 에 없으면 PyG 가 ACM(HGB)을 자동 다운로드합니다.
결과는 `runs/<name>/metrics.json` (test Macro-F1/accuracy, GTN 어텐션, fusion β 등)에 저장됩니다.

클러스터(SLURM):
```bash
sbatch experiments/run_acm.slurm        # baseline×3 + full×3 (seed 0/1/2) 배열
```

## 데이터셋만 바꿔서 같은 실험 돌리기

이 저장소는 **ACM** 만 배선돼 있지만, 데이터셋 교체가 쉽게 설계돼 있습니다:

1. `tda/data/<name>.py` 에 로더 작성 → `(PyG HeteroData, target_node_type)` 반환,
   `tda/data/registry.py` 의 `DATASETS` 에 등록.
2. `configs/_template.json` 을 `configs/<name>.json` 으로 복사하고 `base_relations`
   (타겟에서 시작·종료하는 엣지 타입 시퀀스)와 `han_metapaths` 를 채움.
3. `python -m tda.train --config configs/<name>.json --dataset <name>` 실행.

`configs/toy.json` + `tda/data/toy.py` 가 최소 예시입니다.

## 실험 변형

같은 코드로 config 플래그만 바꿔 실행합니다:

- `use_topology: true/false` (또는 `--no-topology`) — full vs baseline (HAN 단독)
- `gtn.num_channels`, `gtn.num_layers` — 발견 메타패스 수 / 깊이 ablation
- `pdgnn.knn_k`, `pdgnn.hks_K`, `pdgnn.pi_resolution` — 위상 추출 설정
- `seed` — 견고성(분산) 분석

> 범위 메모: 본 저장소는 downstream 을 **HAN 만**, fusion 을 **attention 만** 지원합니다
> (HGT, concat fusion 은 의도적으로 제외).

---

## 충실도 & 가정 (중요)

연구 재현 코드 규약에 따라, 원본과 다른 선택은 모두 명시합니다.

- **PDGNN 은 진짜 neural EPD 근사기**입니다. 정확 EPD(gudhi)는 **학습 라벨로만** 쓰고,
  추론 시 정확 계산은 없습니다. 이 저장소에는 위상을 가장한 단순 그래프 통계("fallback")가
  **없습니다**.
- **GTN 은 타겟 노드 타입에 한정(target-restricted)** 한 버전입니다 — 프로젝트 플랜 §4.1
  ("채널은 한 노드 타입 위의 인접행렬")에 맞춤. 이종 엣지로부터 미리 만든 기저 관계
  (타겟–타겟 1-hop 메타관계 + 동종 엣지 + 항등)에 공식 GTN 의 GTConv/합성을 적용합니다.
- **메타패스 합성**: 플랜 스케치는 forward 에서 채널 자기제곱(`H@H`)으로 적었으나, 공식 GTN
  (레이어마다 새 soft-combo 와의 곱)을 따릅니다(원논문/repo 우선).
- **EPD 필터 = HKS K=3 → 75차원**. 플랜 스케치는 "degree" 라 적었으나, 확립된 파이프라인
  노트(resolution=5 → 75-dim)와 일치하는 HKS 를 채택했습니다.
- **persistence image** 는 표준 PI(Adams et al.)를 순수 numpy 로 재구현 — 원본 cython
  `sg2dgm` 의 수치를 그대로 재현한 것은 아닙니다(공유/Colab 호환 목적).
- **성능 주장은 하지 않습니다.** 표에 나오는 수치는 이 저장소에서 실제로 돌린 결과이며,
  toy 결과는 동작 확인용(sanity)일 뿐 성능 근거가 아닙니다. 선행 연구에서 위상 특징의 평균
  성능 향상은 작고 config 에 민감했고(주된 효과는 분산 감소), 본 파이프라인이 그것을
  넘는다는 증거는 아직 없습니다 — **가설**로만 제시합니다.

## 결과

ACM 실험 결과(test Macro-F1, seed 0/1/2)는 `runs/` 에 저장되며,
요약은 `docs/design.ko.md` 에 정리합니다. (실행 후 채워집니다.)

## 테스트

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
```
