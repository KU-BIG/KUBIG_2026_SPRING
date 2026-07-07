# Topo-HetGNN

**Combining Meta-Path Discovery and Persistent Homology for Heterogeneous Graph Learning**

GTN(메타패스 자동 발견) → PDGNN(EPD 근사) → HAN/RGCN 주입 파이프라인으로, "위상(persistent homology) 특징이 이종 그래프 node classification에 기여하는가"를 통제 실험(noise·class-mix 대조군, paired 검정)으로 검증한 프로젝트.

## 01. Introduction
- MPNN은 local aggregation에 갇혀 그래프의 전역 위상(H0 성분, H1 루프)에 눈멀어 있다. 이를 Persistent homology로 보완
- Persistent homology 정보를 담고 있는 EPD 추출기(PDGNN)는 동종 그래프 전용이므로 GTN이 학습한 meta-path로 이종 → 동종 변환 후 결합한다.

## 02. Methodology
- 3-stage 파이프라인: GTN(채널 그래프 4개) → PDGNN(노드별 75d persistence image) → backbone 주입.
- 실험 매트릭스 = 백본{HAN, RGCN} × 내용{없음, random noise, class-wise mix, real 위상} × 주입{concat, gate}. 10 random seeds,
- 5 데이터셋(acm·dblp·imdb·mag·aifb) 사용.

**주입 방식 1 — concat (위상을 노드 feature의 '내용'으로):**

```math
\tilde{g}_u=\sum_{c=1}^{C}\beta_c\, t_u^{(c)}\in\mathbb{R}^{75},\qquad \beta=\mathrm{softmax}(w_1,\ldots,w_C)
```
```math
\tilde{x}_u=[\,x_u \,\Vert\, \tilde{g}_u\,]\in\mathbb{R}^{F+75},\qquad
W_r\tilde{x}_v=\underbrace{W_r^{(x)}x_v}_{\text{feature}}+\underbrace{W_r^{(g)}\tilde{g}_v}_{\text{topology}}
```

채널 융합(semantic attention β) 후 노드 feature에 이어붙임 → 이웃 집계에서 평균화되며 잡음이 그대로 유입.

**주입 방식 2 — gate (위상을 엣지 메시지의 '밸브'로, PEGN식):**

```math
h'_u=W_0h_u+\sum_{r\in\mathcal{R}}\frac{1}{|N_r(u)|}\sum_{v\in N_r(u)}(W_r h_v)\odot g_{uv}
```
```math
g_{uv}=\sigma\!\left(\mathrm{MLP}\left([\,\tilde{g}_u \,\Vert\, \tilde{g}_v\,]\right)\right)\in(0,1)^d
```

양끝 노드의 위상을 비교해 (0,1) 밸브를 만들고 **집계 전에** 메시지별로 곱함 — 재가중만 가능하므로 해악에 상한이 있음.

## 03. Results

test macro-F1 (mean±std, 10 seeds). dblp HAN은 venue 포함 메타패스 목록 기준.

**RGCN**

| 데이터셋 | base | concat+real | gate+noise | gate+mix | gate+real |
|---|---|---|---|---|---|
| acm | 0.925±0.006 | 0.903±0.006 | 0.924±0.011 | 0.934±0.007 | 0.932±0.010 |
| dblp | 0.934±0.004 | 0.859±0.017 | 0.923±0.007 | 0.962±0.004 | 0.938±0.003 |
| imdb | 0.636±0.004 | 0.566±0.035 | 0.629±0.005 | 0.637±0.005 | 0.640±0.005 |
| mag | 0.104±0.053 | 0.066±0.036 | 0.058±0.017 | 0.091±0.047 | 0.082±0.033 |
| aifb | 0.752±0.018 | 0.643±0.287 | 0.684±0.065 | 0.763±0.013 | 0.752±0.014 |

**HAN**

| 데이터셋 | base | concat+real | gate+noise | gate+mix | gate+real |
|---|---|---|---|---|---|
| acm | 0.895±0.006 | 0.894±0.008 | 0.867±0.009 | 0.884±0.006 | 0.887±0.004 |
| dblp | 0.937±0.005 | 0.938±0.003 | 0.924±0.009 | 0.936±0.007 | 0.934±0.005 |
| imdb | 0.438±0.009 | 0.402±0.015 | 0.464±0.010 | 0.500±0.011 | 0.491±0.017 |
| mag | 0.017±0.012 | 0.023±0.007 | 0.032±0.015 | 0.036±0.015 | 0.037±0.017 |
| aifb | 0.451±0.040 | 0.575±0.145 | 0.585±0.087 | 0.724±0.139 | 0.722±0.144 |

### 1. concat 주입은 homology 정보의 올바른 injection이 아니다
- RGCN 전면 하락(최대 −0.165\*),
- HAN의 상승 사례는 baseline 고착의 구출 효과.
- concat 방식은 구조 상관 문제를 일으킬 수 있다. 구조 space와 Node feature space를 섞어버리기 때문에

### 2. gate(엣지 밸브 σ(MLP([g_u‖g_v])))는 concat의 유해함을 중화한다
- base 동급 복원, 단 이득 창출은 없음.
- gate는 구조 정보를 구조 제어에 사용했기 때문

### 3. gate도 내용물이 진짜 위상일 때만 작동한다
- gate+noise는 전 데이터셋에서 gate+real 미만.
- gate는 구조 정보를 구조 제어에 사용하도록 학습되었기 때문이다.

### 4. 남는 신호는 노드별 위상이 아니라 클래스 수준이다
- gate+mix ≥ gate+real (per-node 정렬의 기여는 0..).
- mix는 class 안에서만 섞기 때문에, 섞은 후에도 남는 정보는 class 수준의 위상 신호뿐이다.
- gate가 class수준의 위상만으로도 real과 동급 이상으로 작동한다

## 4. Conclusion & Future Work
-  위상 특징의 고유 기여는 백본과 무관하게 거의 0.
-  하지만 class의 위상 구조를 담은 high level의 정보는 도움이 됐다 (gate injection 방식에서)
-  백본 간 하락 폭 차이는 잡음 내성(HAN의 attention 필터 vs RGCN의 무필터 평균 집계) 차이일 뿐.
-  향후: link prediction — featureless 그래프에서 유용할 것이라고 예상

# 저장소 구조

```
tda/           코어 패키지 (models/ GTN·PDGNN·HAN·RGCN·Gated*, topology/ EPD·HKS·PI, train.py, gated.py, lp.py)
configs/       데이터셋별 base config (campaign config 는 gen_* 스크립트로 생성)
experiments/   config 생성기(gen_*), SLURM 러너(*.slurm), 캠페인 오케스트레이터(run_*), 집계·검정(agg_*, paired_stats*)
tests/         회귀 테스트 (GTN NaN, topology cache RNG, gated/LP)
```

## Reproduction

```bash
# 환경
conda create -n tlcgnn python=3.9 -y && conda activate tlcgnn
pip install -r requirements.txt        # torch / torch_geometric / gudhi 버전 고정
pytest tests/ -q                       # 회귀 테스트

# 단일 실행 예 (dblp, HAN + GTN-PDGNN 위상, seed 고정)
python -m tda.train --config configs/dblp.json --dataset dblp \
    --data-root <DATA_ROOT> --output-dir runs/demo --seed 312132

# 1) 본표 (a)~(f) 캠페인 — SLURM (seeds: experiments/seeds.txt)
python experiments/gen_full_campaign.py
bash experiments/run_campaign.sh              # baseline·위상 조건
bash experiments/run_d.sh                     # (d) RGCN
bash experiments/run_noise.sh                 # (b1)(e1) random noise
bash experiments/run_class_wise_mixing.sh     # (b2)(e2) class-wise mixing

# 2) 주입 factorial — concat vs gate (고정 GTN+PDGNN 위상)
python experiments/gen_gated.py
MAXGPU=8 bash experiments/run_gated2_campaign.sh

# 3) Link prediction (L1 node-PI / L2 pair-vicinity EPD)
python experiments/gen_lp.py
bash experiments/run_lp_campaign.sh && bash experiments/run_lp2_campaign.sh

# 4) 집계·검정
python experiments/regen_new.py               # 본표 mean±std 집계
python experiments/agg_gated.py && python experiments/agg_lp.py
python experiments/paired_stats2.py           # paired Wilcoxon
```
