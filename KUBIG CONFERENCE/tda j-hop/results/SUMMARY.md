# 새 실험 결과 — 7개 데이터셋 × (a)~(f)

노드 분류 **test Macro-F1, mean±std** (random seed 10개: 15092 88015 238623 251219 312132 500973 588722 661491 792965 825661).

(a)=HAN-only, (c)=HAN+GTN-PDGNN, (d)=RGCN-only, (f)=RGCN+GTN-PDGNN. (b)(e) noisy-topology 는 미실행.

| 데이터셋 | 도메인 | (a) HAN | (b) HAN+noisy | (c) HAN+GTN-PDGNN | (d) RGCN | (e) RGCN+noisy | (f) RGCN+GTN-PDGNN |
|---|---|---|---|---|---|---|---|
| acm | 학술/인용 | 0.895±0.006 (n=10) | —(미실행) | 0.894±0.008 (n=10) | 0.925±0.006 (n=10) | —(미실행) | (미완) (n=0) |
| dblp | 학술/인용 | 0.786±0.013 (n=10) | —(미실행) | 0.862±0.013 (n=10) | 0.934±0.004 (n=10) | —(미실행) | (미완) (n=0) |
| imdb | 영화(멀티라벨) | 0.438±0.009 (n=10) | —(미실행) | 0.450±0.019 (n=10) | 0.636±0.004 (n=10) | —(미실행) | (미완) (n=0) |
| freebase | 지식그래프 | 0.146±0.055 (n=10) | —(미실행) | 0.144±0.051 (n=10) | 0.209±0.108 (n=10) | —(미실행) | (미완) (n=0) |
| mag | 학술(초대형) | 0.017±0.012 (n=10) | —(미실행) | 0.023±0.007 (n=10) | 0.104±0.053 (n=10) | —(미실행) | (미완) (n=0) |
| aifb | RDF/연구기관 | 0.451±0.040 (n=10) | —(미실행) | 0.575±0.145 (n=10) | 0.752±0.018 (n=10) | —(미실행) | (미완) (n=0) |
| yelp | business(멀티라벨) | 0.110±0.028 (n=10) | —(미실행) | 0.091±0.024 (n=10) | 0.055±0.006 (n=10) | —(미실행) | (미완) (n=0) |

진척: **(a) 7/7, (c) 7/7, (d) 7/7, (f) 0/7** 데이터셋 완료 (각 최대 10 seed). (b)(e) noisy-topology 미실행.

## 매핑 / 상태
- (a) HAN only = `a1_baseline`  ·  (c) HAN+GTN-PDGNN = `c2_gtn`  (backbone=han)
- (d) RGCN only = `d_rgcn`  ·  (f) RGCN+GTN-PDGNN = `f_rgcn`  (backbone=rgcn, RGCNConv)
- (b)(e) noisy topological → 정의 확정 필요 (미실행)

원본 per-run: `runs/campaign/<ds>__{a1_baseline,c2_gtn}_s<seed>/metrics.json`

