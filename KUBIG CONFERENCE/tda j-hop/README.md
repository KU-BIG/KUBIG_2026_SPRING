# TDA-HetGNN — 위상 특징이 이종 그래프 노드 분류를 돕는가?

이종(heterogeneous) 그래프에서 **GTN이 자동 발견한 메타패스** 위에 **PDGNN으로 근사한 위상
특징(EPD)** 을 얹어, 두 백본(**HAN**, **RGCN**)의 노드 분류 성능을 비교하는 연구 코드.

> 핵심 질문: (1) 위상 특징이 이종 그래프 학습을 돕는가? (2) 메타패스를 사람이 고르지 않고
> GTN으로 자동 발견해도 되는가? (3) 백본(HAN vs RGCN)에 따라 답이 달라지는가?

## 파이프라인

```
이종 그래프
 └─[GTN]      메타패스(채널) 자동 발견          → 채널별 target–target 인접행렬
     └─[PDGNN]   채널마다 EPD(위상)를 신경망 근사  → 노드별 persistence image
         └─[Semantic Attention Fusion]  채널 위상 특징 융합
             └─[HAN | RGCN]   [원본 feature ⊕ 위상]으로 노드 분류
```

| 모듈 | 출처 | 역할 |
|---|---|---|
| GTN | Yun et al., NeurIPS 2019 | 기저 관계를 가중합·행렬곱으로 합성해 메타패스 채널 자동 발견 |
| PDGNN | Yan et al., NeurIPS 2022 | 확장 persistence diagram(EPD)의 신경망 근사 (필터 = HKS) |
| HAN | Wang et al., WWW 2019 | 메타패스 기반 이종 GNN 백본 |
| RGCN | Schlichtkrull et al., ESWC 2018 | 관계별 weight matrix 백본 (PyG `RGCNConv`) |
| Persistence Image | Adams et al., JMLR 2017 | EPD → 고정 길이 벡터 |

## 실험 설계 — 2 백본 × 3 위상 조건

| | topology 없음 | noise (random) | noise (class-mix) | GTN-PDGNN topology |
|---|:---:|:---:|:---:|:---:|
| **HAN** | **(a)** | **(b1)** | (b2) | **(c)** |
| **RGCN** | **(d)** | **(e1)** | (e2) | (f) |

- **(a)/(d) baseline** — node feature만으로 분류 (위상 없음).
- **(b1)/(e1) random noise** — 위상 자리에 **같은 차원(res²×K)의 랜덤 가우시안**을 concat.
  GTN/PDGNN을 아예 안 돌림 → 성능 변화가 *단순 차원 추가* 때문인지 격리하는 대조군.
- **(b2)/(e2) class-wise mixing** — **실제 GTN-PDGNN 위상**을 계산하되, **같은 class·같은
  split(train/val/test) 안에서 노드 간 shuffle** (`topology_mode=class_wise_mixing`).
  class-level 위상 분포는 보존하고 *per-node 위상↔노드 매칭만* 파괴 → "노드별 위상 정보가
  class 정보 이상으로 기여하는가"를 격리. (cf. within-class feature shuffle,
  [Lee et al., ICML 2024](https://arxiv.org/abs/2402.04621); conditional permutation
  importance, Strobl et al. 2008)
- **(c)/(f) real topology** — GTN이 발견한 메타패스에서 PDGNN이 근사한 EPD 위상을 concat.

해석 논리: **real(c/f)이 random(b1/e1)과 class-mix(b2/e2)를 둘 다 이겨야** GTN-PDGNN 위상이
per-node 수준에서 진짜 의미 있다고 주장 가능. real≈class-mix>random이면 위상 가치는
class-level 분포뿐(개별 노드 매칭은 잉여)이라는 결론.

`backbone ∈ {han, rgcn}` config 플래그로 백본 전환. 현재 **(a)(b1)(c)(d)(e1) 완료**, (b2)(e2)(f) 대기.

## 결과 (7개 데이터셋 · test Macro-F1 · random seed 10개)

미실행 조건((b2)(e2)(f))은 `–`.

| 데이터셋 | 도메인 | (a) HAN | (b1) +noise | (b2) +mix | (c) +위상 | (d) RGCN | (e1) +noise | (e2) +mix | (f) +위상 |
|---|---|---|---|---|---|---|---|---|---|
| acm | 학술/인용 | 0.895 | 0.887 | – | 0.894 | **0.925** | 0.919 | – | – |
| dblp | 학술/인용 | 0.786 | 0.776 | – | **0.862** | **0.934** | 0.930 | – | – |
| imdb | 영화(멀티라벨) | 0.438 | 0.441 | – | 0.450 | **0.636** | 0.579 | – | – |
| freebase | 지식그래프 | 0.146 | 0.160 | – | 0.144 | **0.209** | 0.158 | – | – |
| mag | 학술(초대형) | 0.017 | 0.026 | – | 0.023 | **0.104** | 0.090 | – | – |
| aifb | RDF | 0.451 | 0.478 | – | **0.575** | **0.752** | 0.673 | – | – |
| yelp | business(멀티라벨) | 0.110 | 0.094 | – | 0.091 | 0.055 | 0.056 | – | – |

- **noise 대조 (b1 vs a, e1 vs d)**: random noise는 baseline과 비슷하거나 아래 (RGCN에선 imdb/freebase/aifb에서 뚜렷이 ↓) → 단순 차원 추가는 이득 없음.
- **위상 효용 (c vs b1)**: feature 약한 데이터셋에서 real 위상이 noise를 뚜렷이 이김 (dblp +0.086, aifb +0.097) → 위상 신호 실재. feature 강하면 ≈0.
- **백본 (d vs a)**: RGCN이 HAN을 6/7에서 크게 상회 (yelp 제외).
- **MAG·yelp 절대값 주의**: macro-F1이 낮은 건 metric 특성(MAG=349클래스 subsample 평균, yelp=희귀 멀티라벨·featureless). accuracy는 MAG 0.28, yelp 0.87(RGCN)로 정상 학습 — 이 둘은 데이터셋 내 조건 간 Δ로만 해석.

전체 표·진행률·매핑은 [`results/SUMMARY.md`](results/SUMMARY.md), 데이터셋 특징은
[`results/DATASETS.md`](results/DATASETS.md).

## 시각화

```bash
python experiments/visualize_pipeline.py <dataset>   # results/figures/<dataset>/ 에 저장
```

세 가지를 생성합니다 (랜덤 노드 100개씩, near-clique degenerate 노드는 자동 제외):

**1. 메타패스 구성** — GTN이 기저관계를 학습 가중치로 합성해 메타패스를 만드는 과정.

![meta-path construction](docs/figures/metapath_construct.png)

**2. 메타패스 결과** (`metapath/` 100개) — 메타패스 이웃의 클래스 묶음 + same-class homophily.

![meta-path result](docs/figures/metapath_result.png)

**3. EPD 생성** (`epd/` 100개) — 노드 ego + HKS 필터 → persistence diagram → persistence image.

![EPD process](docs/figures/epd_process.png)

## 설치 & 실행

```bash
conda create -n tda python=3.9 -y && conda activate tda
pip install -r requirements.txt        # torch / torch_geometric 는 플랫폼 휠 권장
pip install -e .

# (a) HAN 단독
python -m tda.train --config configs/acm.json --dataset acm --no-topology --output-dir runs/a --seed 0
# (c) HAN + GTN-PDGNN 위상
python -m tda.train --config configs/acm.json --dataset acm --output-dir runs/c --seed 0
# (d)/(f) RGCN — config 생성 후 backbone=rgcn config 사용
python experiments/gen_rgcn.py
python -m tda.train --config configs/campaign/acm__d_rgcn.json --dataset acm --output-dir runs/d --seed 0
```

결과는 `runs/<name>/metrics.json`. 클러스터 일괄 실행은 `experiments/run_campaign.slurm` (SLURM 배열,
`%N`으로 동시 GPU 제한). 테스트: `env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q`.

## 새 데이터셋 추가

1. `tda/data/<name>.py` 에 `(PyG HeteroData, target_type)` 로더 작성 → `tda/data/registry.py` 의 `DATASETS` 등록.
2. `configs/<name>.json` 에 `base_relations`(GTN 기저 관계) + `han_metapaths`(HAN 메타패스) 정의.
3. `python -m tda.train --config configs/<name>.json --dataset <name>` 실행.

최소 예시: `tda/data/toy.py` + `configs/toy.json`. 데이터셋별 튜닝값은 전부 config에 있음.

## 코드 구조

```
tda/
  models/      gtn.py  pdgnn.py  han.py  rgcn.py  fusion.py
  topology/    hks.py  epd.py  persistence_image.py  cache.py(재실험용 위상 캐시)
  data/        registry.py + 데이터셋 로더들
  train.py     staged 학습 드라이버 (backbone ∈ {han, rgcn})
  viz.py       per-run 시각화(기본 OFF)
experiments/   gen_rgcn.py  run_campaign.slurm  regen_new.py  visualize_pipeline.py
configs/       데이터셋별 하이퍼파라미터
results/       SUMMARY.md(결과표)  DATASETS.md(데이터셋 특징)
```

## 참고

GTN (NeurIPS'19) · PDGNN (NeurIPS'22) · HAN (WWW'19) · RGCN (ESWC'18) · Persistence Image (JMLR'17).
설계·한계 상세는 [`docs/design.ko.md`](docs/design.ko.md).
