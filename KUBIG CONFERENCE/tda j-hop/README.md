# Topo-HetGNN

**Combining Meta-Path Discovery and Persistent Homology for Heterogeneous Graph Learning**

GTN(메타패스 자동 발견) → PDGNN(EPD 근사) → HAN/RGCN 주입 파이프라인

**위상(persistent homology) 특징이 이종 그래프 node classification에 기여하는가**를 noise/class-mix 대조 실험으로 검증한 프로젝트이다.

## 01. Introduction

**배경**
- 실세계 데이터(학술, 지식그래프, 리뷰 등)는 여러 타입의 노드와 관계가 섞인 heterogeneous graph이다.
- 대표 모델(HAN,RGCN 등 MPNN 계열)은 이웃 feature를 aggregate하는 방식 → **수용 범위가 local(layer 수 × hop)에 갇힘**
- 같은 이웃 구조를 가진 두 노드라도 전역 위상(연결 성분 H0, 루프 H1)은 다를 수 있음 — 이 정보가 feature에 담기지 않음 (**topology blindness**)

**접근**
- Persistent homology: filtration 과정에서 위상 특징의 생성(birth)과 소멸(death)을 추적 → Persistent homology를 이미지 벡터(persistence image)로 요약
- 단 EPD 추출기(PDGNN)는 homogeneous 전용 → **GTN이 학습한 meta-path로 heterogeneous → homogeneous 서브그래프 변환** 후 결합
- 전체 파이프라인: **GTN → PDGNN → backbone(HAN/RGCN)**

**연구 질문**
- **Q1.** GTN-PDGNN으로 추출한 homology feature를 이종 GNN에 추가하면 node classification 성능이 향상되는가?
- **Q2.** 향상이 있다면 진짜 위상 신호 때문인가, feature 차원 증가 효과인가? → noise·class-mix 대조군으로 검증
- **Q3.** 위상 feature의 기여는 backbone 구조(HAN vs RGCN)에 따라 달라지는가?

## 02. Methodology
- 3-stage 파이프라인: GTN(채널 그래프 4개) → PDGNN(노드별 75d persistence image) → backbone 주입.
- 실험 매트릭스 = 백본{HAN, RGCN} × 내용{없음, random noise, class-wise mix, real 위상} × 주입{concat, gate}. 10 random seeds,
- 5 데이터셋(acm·dblp·imdb·mag·aifb) 사용.

**주입 방식 1 : concat (위상을 노드 feature의 내용으로):**

```math
\tilde{g}_u=\sum_{c=1}^{C}\beta_c\, t_u^{(c)}\in\mathbb{R}^{75},\qquad \beta=\mathrm{softmax}(w_1,\ldots,w_C)
```
```math
\tilde{x}_u=[\,x_u \,\Vert\, \tilde{g}_u\,]\in\mathbb{R}^{F+75},\qquad
W_r\tilde{x}_v=\underbrace{W_r^{(x)}x_v}_{\text{feature}}+\underbrace{W_r^{(g)}\tilde{g}_v}_{\text{topology}}
```

채널 융합(semantic attention β) 후 노드 feature에 이어붙임 → 이웃 집계에서 평균화되며 잡음이 그대로 유입할 수가 있다

**주입 방식 2 : gate (위상을 엣지 메시지의 gate로, PEGN식):**

```math
h'_u=W_0h_u+\sum_{r\in\mathcal{R}}\frac{1}{|N_r(u)|}\sum_{v\in N_r(u)}(W_r h_v)\odot g_{uv}
```
```math
g_{uv}=\sigma\!\left(\mathrm{MLP}\left([\,\tilde{g}_u \,\Vert\, \tilde{g}_v\,]\right)\right)\in(0,1)^d
```

양끝 노드의 위상을 비교해 (0,1) 밸브를 만들고 **집계 전에** 메시지별로 곱함 

## 03. Results

test macro-F1 (mean±std, 10 seeds).
결과분석 (https://www.notion.so/3942bc41882d804b8a59e44feba7d366) 참고.
### 1. concat 주입은 homology 정보의 올바른 injection이 아니다

| 데이터셋 | RGCN: base → concat | HAN: base → concat |
|---|---|---|
| acm | 0.925 → 0.903 | 0.895 → 0.894 |
| dblp | 0.934 → **0.859** | 0.937 → 0.938 |
| imdb | 0.636 → **0.566** | 0.438 → **0.402** |
| mag | 0.104 → 0.066 | 0.017 → 0.023 |
| aifb | 0.752 → **0.643** | 0.451 → 0.575 (구출) |

- RGCN 전면 하락(최대 −0.165\*),
- HAN의 상승 사례는 baseline 고착의 구출 효과.
- concat 방식은 구조 상관 문제를 일으킬 수 있다. 구조 space와 Node feature space를 섞어버리기 때문에

### 2. gate(엣지 밸브 $`g_{uv}=\sigma(\mathrm{MLP}([\tilde{g}_u \Vert \tilde{g}_v]))`$)는 concat의 유해함을 중화한다

| 데이터셋 | RGCN: concat → gate (base) | HAN: concat → gate (base) |
|---|---|---|
| acm | 0.903 → **0.932** (0.925) | 0.894 → 0.887 (0.895) |
| dblp | 0.859 → **0.938** (0.934) | 0.938 → 0.934 (0.937) |
| imdb | 0.566 → **0.640** (0.636) | 0.402 → **0.491** (0.438) |
| mag | 0.066 → 0.082 (0.104) | 0.023 → 0.037 (0.017) |
| aifb | 0.643 → **0.752** (0.752) | 0.575 → 0.722 (0.451) |

- base 동급 복원, 단 이득 창출은 없음.
- gate는 구조 정보를 구조 제어에 사용했기 때문

### 3. gate도 내용물이 진짜 위상일 때만 작동한다

| 데이터셋 | RGCN: gate+real / gate+noise | HAN: gate+real / gate+noise |
|---|---|---|
| acm | **0.932** / 0.924 | **0.887** / 0.867 |
| dblp | **0.938** / 0.923 | **0.934** / 0.924 |
| imdb | **0.640** / 0.629 | **0.491** / 0.464 |
| mag | **0.082** / 0.058 | **0.037** / 0.032 |
| aifb | **0.752** / 0.684 | **0.722** / 0.585 |

- gate+noise는 전 데이터셋에서 gate+real 미만.
- gate는 구조 정보를 구조 제어에 사용하도록 학습되었기 때문이다.

### 4. 남는 신호는 노드별 위상이 아니라 클래스 수준이다

| 데이터셋 | RGCN: gate+real / gate+mix | HAN: gate+real / gate+mix |
|---|---|---|
| acm | 0.932 / **0.934** | **0.887** / 0.884 |
| dblp | 0.938 / **0.962** | 0.934 / **0.936** |
| imdb | **0.640** / 0.637 | 0.491 / **0.500** |
| mag | 0.082 / **0.091** | **0.037** / 0.036 |
| aifb | 0.752 / **0.763** | 0.722 / **0.724** |

- gate+mix ≥ gate+real (per-node 정렬의 기여는 0..).
- mix는 class 안에서만 섞기 때문에, 섞은 후에도 남는 정보는 class 수준의 위상 신호뿐이다.
- gate가 class수준의 위상만으로도 real과 동급 이상으로 작동한다

## 4. Conclusion & Future Work


- **Q1. 위상 feature 추가로 NC 성능이 향상되는가?** → **향상 되기는 함** 하지만 진짜 정확한 위상 때문은 아님. injection 방식으로 concat은 유해, gate는 그 유해함을 중화할 뿐 base 대비 이득 없음.
- **Q2. 진짜 위상 신호 때문인가?** → **아니오.** 차원 효과는 아니지만(noise 대조), 남는 신호는 per-node(진짜) 위상이 아닌 class 수준 신호
- **Q3. backbone에 따라 달라지는가?** → **아니오.** 고유 기여는 HAN·RGCN 모두 ≈ 0. 하락 폭 차이는 잡음 내성 차이일 뿐.

-  위상 특징의 고유 기여는 백본과 무관하게 거의 0.
-  하지만 class의 위상 구조를 담은 high level의 정보는 도움이 됐다 (gate injection 방식에서)
-  백본 간 하락 폭 차이는 잡음 내성(HAN의 attention 필터 vs RGCN의 무필터 평균 집계) 차이일 뿐.
-  향후: link prediction — featureless 그래프에서 유용할 것이라고 예상

# 저장소 구조

```
tda/           코어 패키지 (models/ GTN·PDGNN·HAN·RGCN·Gated*, topology/ EPD·HKS·PI, train.py, gated.py, lp.py)
configs/       데이터셋별 base config (campaign config 는 gen_* 스크립트로 생성)
experiments/   config 생성기(gen_*), seed 목록(seeds.txt), 집계·검정(regen_new, agg_*, paired_stats2)
tests/         회귀 테스트 (GTN NaN, topology cache RNG, gated/LP)
```

## Reproduction

CUDA GPU 1개 기준. `experiments/seeds.txt`의 10개 seed로 config 하나씩 순차 실행하고, 완료된 run(`metrics.json` 존재)은 건너뛴다.

```bash
# 환경
conda create -n tlcgnn python=3.9 -y && conda activate tlcgnn
pip install -r requirements.txt        # torch / torch_geometric / gudhi 버전 고정
pytest tests/ -q                       # 회귀 테스트

export DATA_ROOT=<DATA_ROOT>           # 데이터셋 루트

# 단일 실행 예 (dblp, HAN + GTN-PDGNN 위상, seed 고정)
python -m tda.train --config configs/dblp.json --dataset dblp \
    --data-root "$DATA_ROOT" --output-dir runs/demo --seed 312132

# 공통 러너: run <모듈> <config> <출력루트>  — config 하나 × 10 seed
run() {
  local NAME DS; NAME=$(basename "$2" .json)
  DS=$(python -c "import json;print(json.load(open('$2'))['dataset'])")
  while read -r SEED; do
    OUT=$3/${NAME}_s${SEED}
    [ -f "$OUT/metrics.json" ] || python -m "$1" --config "$2" --dataset "$DS" \
        --data-root "$DATA_ROOT" --output-dir "$OUT" --seed "$SEED"
  done < experiments/seeds.txt
}

# 1) 본표 (a)~(f): 7 ds × 8조건 × 10 seed → runs/campaign
python experiments/gen_full_campaign.py       # (a) a1_baseline · (c) c2_gtn
python experiments/gen_noise.py               # (b1) b1_noise   · (e1) d1_noise
python experiments/gen_class_wise_mixing.py   # (b2) b2_mix     · (e2) e2_mix
python experiments/gen_rgcn.py                # (d) d_rgcn      · (f) f_rgcn
for ds in acm dblp imdb freebase mag aifb yelp; do
  for c in a1_baseline b1_noise b2_mix c2_gtn d_rgcn d1_noise e2_mix f_rgcn; do
    run tda.train configs/campaign/${ds}__${c}.json runs/campaign
  done
done

# 2) 주입 factorial (concat vs gate): 5 ds × 14조건 × 10 seed → runs/gated
#    real/mix 는 고정 GTN+PDGNN 위상(gt2_/gh2_), base/noise 는 위상 무관(gt_/gh_)
python experiments/gen_gated.py
for ds in acm dblp imdb mag aifb; do
  for c in gt_base gt2_cat_real gt_cat_noise gt2_cat_mix gt2_gate_real gt_gate_noise gt2_gate_mix \
           gh_base gh2_cat_real gh_cat_noise gh2_cat_mix gh2_gate_real gh_gate_noise gh2_gate_mix; do
    run tda.gated configs/campaign/${ds}__${c}.json runs/gated
  done
done

# 3) Link prediction: L1 node-PI(3 ds) + L2 pair-vicinity EPD(7 ds) → runs/lp
python experiments/gen_lp.py
for ds in acm dblp aifb; do
  for c in lp_a lp_b1 lp_c lp_m; do run tda.lp configs/campaign/${ds}__${c}.json runs/lp; done
done
run tda.lp configs/campaign/dblp__lp_cx.json runs/lp
for ds in acm dblp imdb freebase mag aifb yelp; do
  for c in lp2_base lp2_real lp2_noise lp2_mix; do run tda.lp configs/campaign/${ds}__${c}.json runs/lp; done
done

# 4) 집계·검정 → results/
python experiments/regen_new.py               # 본표 (a)~(f) mean±std
python experiments/agg_gated.py               # factorial 표
python experiments/agg_lp.py                  # LP 표
python experiments/paired_stats2.py           # paired Wilcoxon 검정
```

> dblp의 HAN 결과는 metapath 목록에 venue를 포함한 것 (`configs/dblp.json`의 `han_metapaths`를 `["APA","APTPA","APVPA"]`로 교체 후 실행).
