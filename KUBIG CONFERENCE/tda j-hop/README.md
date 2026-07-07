# Topo-HetGNN

**Combining Meta-Path Discovery and Persistent Homology for Heterogeneous Graph Learning**

GTN(메타패스 자동 발견) → PDGNN(EPD 근사) → HAN/RGCN 주입 파이프라인으로, "위상(persistent homology) 특징이 이종 그래프 node classification에 기여하는가"를 통제 실험(noise·class-mix 대조군, paired 검정)으로 검증한 프로젝트.

## 발표 개요

**01. Introduction** — MPNN은 local aggregation에 갇혀 그래프의 전역 위상(H0 성분·H1 루프)에 눈멀어 있다. Persistent homology로 보완하되, EPD 추출기(PDGNN)는 동종 그래프 전용이므로 GTN이 학습한 메타패스로 이종→동종 변환 후 결합한다.

**02. Methodology** — 3-stage 파이프라인: GTN(채널 그래프 4개) → PDGNN(노드별 75d persistence image) → backbone 주입. 실험 매트릭스 = 백본{HAN, RGCN} × 내용{없음, random noise, class-wise mix, real 위상} × 주입{concat, gate}. 10 random seeds, paired Wilcoxon. 5 데이터셋(acm·dblp·imdb·mag·aifb; dblp HAN은 venue 포함 메타패스 목록 사용).

**03. Results**
1. concat 주입은 homology 정보의 올바른 injection이 아니다 — RGCN 전면 하락(최대 −0.165\*), HAN의 상승 사례는 baseline 고착의 구출 효과.
2. gate(엣지 밸브 σ(MLP([g_u‖g_v])))는 concat의 유해함을 중화한다 — base 동급 복원, 단 이득 창출은 없음.
3. gate도 내용물이 진짜 위상일 때만 작동한다 — gate+noise는 전 데이터셋에서 gate+real 미만.
4. 남는 신호는 노드별 위상이 아니라 클래스 수준이다 — gate+mix ≥ gate+real (per-node 정렬 기여 0).

**04. Conclusion & Future Work** — 위상 특징의 고유 기여는 백본과 무관하게 ≈0. 백본 간 하락 폭 차이는 잡음 내성(HAN의 attention 필터 vs RGCN의 무필터 평균 집계) 차이일 뿐. 향후: link prediction — featureless 그래프에서 pair-vicinity EPD가 +0.07~0.09 AUC(win 100%)로 유일하게 유효.

## 저장소 구조

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

재현 확인용 기대값(10 seeds, mean±std): dblp HAN(venue 포함) base **0.937±0.005** / gate+real 0.934±0.005 · RGCN(d) 0.934±0.004 · imdb concat −0.082\* · aifb concat −0.165\*, gate ±0.000 · LP freebase +pair-PI **0.873**(base 0.783) · yelp **0.959**(base 0.890).
