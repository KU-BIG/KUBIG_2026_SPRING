# TDA: HAN + GTN + PDGNN 파이프라인

## 이게 뭐야

이종(heterogeneous) 그래프에서 **위상 특징(persistent homology, EPD)** 을 자동으로 뽑아
노드 분류에 쓰는 하나의 파이프라인입니다.

```
이종 그래프
  └─[GTN]    메타패스를 자동 발견 (사람이 안 고름)
      └─[PDGNN]   발견된 메타패스마다 EPD(위상)를 신경망으로 근사 → per-node 75차원
          └─[Semantic Attention Fusion]  채널별 위상 특징을 융합
              └─[HAN]   원본 특징 ⊕ 위상 특징으로 노드 분류
```

핵심 질문: **위상 특징이 이종 그래프 학습을 돕는가? 그리고 그걸 가능케 하는 메타패스 선택을
자동화(GTN)할 수 있는가?** — 기존엔 메타패스를 사람이 직접 골랐습니다(예: PAP, PSP).

구성 요소: GTN(Yun et al., NeurIPS 2019) · PDGNN(Yan et al., NeurIPS 2022, neural EPD 근사기) ·
HAN(Wang et al., WWW 2019) · Persistence Image(Adams et al., 2017).
자세한 설계와 한계는 [`docs/design.ko.md`](docs/design.ko.md).

## 설치 & 실행

```bash
conda create -n tda python=3.9 -y && conda activate tda
pip install -r requirements.txt          # torch / torch_geometric 는 플랫폼 휠 권장
pip install -e .

# 전체 파이프라인 (GTN + PDGNN + HAN, attention fusion)
python -m tda.train --config configs/acm.json --dataset acm --output-dir runs/c2 --seed 0
# baseline (HAN 단독)
python -m tda.train --config configs/acm.json --dataset acm --output-dir runs/a1 --seed 0 --no-topology
```

데이터가 없으면 PyG 가 ACM(HGB)을 자동 다운로드합니다. 결과는 `runs/<name>/metrics.json`.
클러스터 + **ablation 토글**: `bash experiments/run_acm.sh off`(메인만) / `bash experiments/run_acm.sh on`(메인 + D1~D6 ablation).
테스트: `env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q`

## 다른 데이터셋 쓰기

**8개 데이터셋**(아래 결과 표)이 이미 배선돼 있고, 새 데이터셋도 **교체가 쉽게** 설계돼 있습니다. 3단계:

1. **로더 작성** — `tda/data/<name>.py` 에 `(PyG HeteroData, target_node_type)` 를 반환하는
   함수를 만들고 `tda/data/registry.py` 의 `DATASETS` 에 등록.
   ```python
   # tda/data/registry.py
   DATASETS = {"acm": load_acm, "toy": load_toy, "<name>": load_<name>}
   ```
2. **config 작성** — `configs/_template.json` 을 `configs/<name>.json` 으로 복사해 채움.
   핵심은 두 개:
   - `base_relations`: 타겟에서 시작·종료하는 엣지 타입 시퀀스(GTN 의 기저 관계). 각 triple 은
     `HeteroData` 의 edge_type 이름과 일치해야 함.
   - `han_metapaths`: HAN 이 쓸 메타패스 그래프.
   ```json
   "base_relations": {
     "MAM": [["movie","to","actor"], ["actor","to","movie"]],
     "MDM": [["movie","to","director"], ["director","to","movie"]]
   },
   "han_metapaths": ["MAM", "MDM"]
   ```
3. **실행** — `python -m tda.train --config configs/<name>.json --dataset <name>`

데이터셋마다 **튜닝해야 하는 값은 전부 config 에** 있습니다(GTN 채널/깊이, PDGNN knn_k/hks_K,
PI 범위/σ, HAN hidden/heads, lr 등). 무엇이 데이터셋별로 바꿔도 되는 값이고 무엇이 고정인지는
[`docs/design.ko.md` §4b](docs/design.ko.md) 의 "충실(고정) vs 설계(유연)" 표 참고.
최소 예시는 `tda/data/toy.py` + `configs/toy.json`.

## 전체 실험(A~D) 한 번에 돌리기

A~D 를 **따로따로 입력할 필요 없습니다** — 전부 **config 플래그**로 정의되고 생성기가 자동으로
만듭니다. 한 명령으로 12개 데이터셋 × 전체 A~D(× seed 0/1/2)를 실행:

```bash
bash experiments/run_campaign.sh   # config 자동생성 + SLURM 청크 제출(GPU 포화) + 집계
```

실험 매트릭스 (각 설정 = config 한 줄, `experiments/gen_full_campaign.py` 참조):

| 그룹 | 설정 | config 플래그 |
|------|------|---------------|
| A1 | HAN 단독 (baseline) | `use_topology=false` |
| A3 | GTN 단독 (EPD 없음) | C2 run 안에 `gtn_only_*` 로 기록 |
| B2 | manual 메타패스 + EPD | `topology_source=manual` |
| C2 | GTN + PDGNN + HAN (메인) | `topology_source=gtn` |
| D2 | MIN 집계 제거 | `pdgnn.agg=sum` |
| D3 | 채널수 {2, 8} | `gtn.num_channels` |
| D4 | GTN 깊이 {1, 3} | `gtn.num_layers` |
| D5 | random 메타패스 | `topology_source=random` |
| 진단 | topology-only / permutation | `node_features=off` / `permute_topology` |

(D1=no-kNN 은 ego 폭증으로 비현실적이라 제외.) 끝나면 `results/SUMMARY.md` 에 mean±std 로 모입니다.
단일 데이터셋만: `python -m tda.train --config configs/<ds>.json --dataset <ds> [--no-topology|--topology-source ...]`.

## 여러 도메인 결과 (8개 데이터셋)

노드 분류 test Macro-F1, baseline(HAN 단독) vs full(GTN+PDGNN+HAN, attention). 원본은 `results/`.

| 데이터셋 | 도메인 | baseline | full | Δ |
|----------|--------|----------|------|---|
| ACM | 학술/인용 | 0.9000 | 0.8961 | −0.0039 |
| DBLP | 학술/인용 | 0.7991 | 0.8737 | **+0.0746** |
| DBLP_pyg | 학술(PyG판) | 0.9341 | 0.9323 | −0.0018 |
| AMiner | 학술(대형·subsample, featureless) | 0.3535 | 0.4024 | **+0.0489** |
| ogbn-mag | 학술(대형·subsample) | 0.0155 | 0.0424 | +0.0268 |
| IMDB | 영화(멀티라벨) | 0.4422 | 0.4296 | −0.0126 |
| IMDB_pyg | 영화(단일라벨) | 0.5125 | 0.5200 | +0.0075 |
| Freebase | 지식그래프(featureless) | 0.1258 | 0.1286 | +0.0027 |

**핵심:** 위상(full)의 효용은 **node feature 가 약하거나 sparse 한 데이터셋에서 크다** —
DBLP +0.075, AMiner(featureless) +0.049, MAG +0.027. 반대로 feature 가 이미 강하면 ≈null —
ACM −0.004, DBLP_pyg(baseline 0.93) −0.002. (Freebase·MAG 는 featureless / 349클래스·subsample
이라 절대값 자체가 낮음 — 비교는 Δ 로.) RCDD(금융)는 PyG 다운로드 링크가 404 라 제외.

## ACM 을 썼을 때의 결과 예시 (ablation 포함)

ACM(paper 3 클래스, Macro-F1), seed 0/1/2 평균 (성능 주장이 아니라 실측치). 메인 + ablation 통합:

| 실험 | test Macro-F1 | 한 줄 의미 |
|------|----------------|-----------|
| A1  HAN 단독 (baseline) | 0.8892 ± 0.0113 | 위상 없는 기준선 |
| A3  GTN 단독 (EPD 없음) | 0.8939 ± 0.0088 | 메타패스만 자동발견 |
| B2  manual 메타패스(PAP/PSP) + PDGNN-EPD | 0.8977 ± 0.0031 | 사람이 고른 메타패스 위 EPD |
| **C2  GTN + PDGNN + HAN (attention)** | **0.8958 ± 0.0035** | **전체 파이프라인 (메인)** |
| D1  kNN 희소화 제거 (PDGNN 앞) | 측정 불가 (>2h 미완) | kNN 없으면 ego 폭증 → 비현실적, 즉 **kNN 필수** |
| D2  MIN 집계 제거 (SUM 만) | 0.8988 | MIN 빼도 무방 (오히려 근소 ↑) |
| D3  num_channels 2 / 4 / 8 | 0.8852 / 0.8958 / 0.8951 | 채널 4가 적당, 2는 부족 |
| D4  GTN depth 1 / 2 / 3 | 0.8965 / 0.8958 / 0.8891 | 얕아도 충분, 깊이 3은 하락 |
| D5  random 메타패스 (학습 없음) | 0.8887 | GTN(0.896) > 랜덤 → 학습이 의미 있음 |
| D6  manual(B2) vs GTN(C2) | 0.8977 vs 0.8958 | GTN이 수동 선택을 못 넘음 |

진단:
- **topology-only** (노드 특징 끄고 위상만): **0.7436** — 랜덤(0.33) 훨씬 위 → 위상에 신호가 있음.
- **permutation** (test 때 위상 행을 섞음): C2 0.8958 → 0.8296 (**Δ +0.066**) → 모델이 위상을 실제로 사용함.

**정직한 해석:** 위상 특징은 신호를 담고 모델이 실제로 쓰지만(위 진단), node feature 가 이미 강해서 **강한 HAN baseline 을 뚜렷이 넘지는 못한다**(B2·C2 ≈ A1, seed 분산 내). 다만 위상 변형은 **분산이 더 작다**(A1 은 seed 1 에서 0.8735 로 출렁이나 B2/C2 는 0.89~0.90 유지) 

- 선행 연구의 "위상의 주효과는 평균이 아니라 분산(robustness)" 와 같은 방향. 단 **seed 3개**라 분산 주장은 통계적으로 약하다.
  
- D1(no-kNN)은 ego 그래프가 커져 실행이 매우 오래 걸린다.

범위·한계는 [`docs/design.ko.md` §8](docs/design.ko.md).

- end-to-end 학습(C4)은 미구현(future work): GTN soft 인접 → 이산 그래프(kNN/ego) 단계 / 미분 불가라 task loss 가 GTN 까지 역전파되지 않아 staged 학습만 했습니다.
