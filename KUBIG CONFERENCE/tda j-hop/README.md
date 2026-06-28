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

ACM 만 배선돼 있지만 **데이터셋 교체가 쉽게** 설계돼 있습니다. 3단계:

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

## ACM 을 썼을 때의 결과 예시

ACM(paper 3 클래스, Macro-F1), seed 0/1/2 평균 (성능 주장이 아니라 실측치):

| 설정 | test Macro-F1 |
|------|----------------|
| A1  HAN 단독 (baseline) | 0.8892 ± 0.0113 |
| A3  GTN 단독 (EPD 없음) | 0.8939 ± 0.0088 |
| B2  manual 메타패스(PAP/PSP) + PDGNN-EPD | 0.8977 ± 0.0031 |
| **C2  GTN + PDGNN + HAN (attention)** | **0.8958 ± 0.0035** |

진단:
- **topology-only** (노드 특징 끄고 위상만): **0.7436** — 랜덤(0.33) 훨씬 위 → 위상에 신호가 있음.
- **permutation** (test 때 위상 행을 섞음): C2 0.8958 → 0.8296 (**Δ +0.066**) → 모델이 위상을
  실제로 사용함.

**정직한 해석:** 위상 특징은 신호를 담고 있고 모델이 실제로 쓰지만(위 진단), node feature 가
이미 강해서 **강한 HAN baseline 을 뚜렷이 넘지는 못한다**(B2·C2 ≈ A1, seed 분산 내). 다만
위상 변형은 **분산이 더 작다**(A1 은 seed 1 에서 0.8735 로 출렁이나 B2/C2 는 0.89~0.90 유지) —
선행 연구의 "위상의 주효과는 평균이 아니라 분산(robustness)" 와 같은 방향. 단 **seed 3개**라
분산 주장은 통계적으로 약하다. GTN-학습 메타패스(C2)는 random(0.8887)보다 낫지만 manual(B2)을
넘지는 못한다.

Ablation (C2 기준 변형, test Macro-F1):

| 실험 | 변형 | 결과 | 한 줄 의미 |
|------|------|------|-----------|
| D1 | kNN 희소화 제거 (PDGNN 앞) | _(실행 중)_ | 희소화 효과 측정용 (없으면 저하 예상) |
| D2 | MIN 집계 제거 (SUM 만) | 0.8988 | MIN 빼도 무방 (오히려 근소 ↑) |
| D3 | num_channels 2 / 4 / 8 | 0.8852 / 0.8958 / 0.8951 | 채널 4가 적당, 2는 부족 |
| D4 | GTN depth 1 / 2 / 3 | 0.8965 / 0.8958 / 0.8891 | 얕아도 충분, 깊이 3은 하락 |
| D5 | random 메타패스 (학습 없음) | 0.8887 | GTN(0.896) > 랜덤 → 학습이 의미 있음 |
| D6 | manual(B2) vs GTN(C2) | 0.8977 vs 0.8958 | GTN이 수동 선택을 못 넘음 |

> D1(no-kNN)은 희소화를 빼면 ego 그래프가 커져 **실행이 매우 오래 걸립니다** (그래서 표가 비어 있음).

범위·한계는 [`docs/design.ko.md` §8](docs/design.ko.md).

> end-to-end 학습(C4)은 미구현(future work): GTN soft 인접 → 이산 그래프(kNN/ego) 단계가
> 미분 불가라 task loss 가 GTN 까지 역전파되지 않아 staged 학습만 했습니다.
