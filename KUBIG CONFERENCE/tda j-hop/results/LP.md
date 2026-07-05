# LP 결과 — 위상 특징 × link prediction (test AUC, mean±std)

설계: docs/lp_design.md. 예측 대상 = 주 타겟-타겟 관계 엣지, 고정 split, encoder=RGCN. L1=node-PI encoder concat(스캔, 5 seed) / L2=pair-vicinity EPD decoder concat(TLC-GNN 식, 10 seed).

## Level 1 — node-PI (encoder 입력 concat)

| 데이터셋 | base | +noise | +mix(node) | +node-PI | +node-PI(관계제외 c') |
|---|---|---|---|---|---|
| acm | 0.894±0.016 (n=5) | 0.900±0.009 (n=5) | 0.902±0.011 (n=5) | 0.894±0.014 (n=5) | – |
| dblp | 0.829±0.008 (n=5) | 0.854±0.006 (n=5) | 0.832±0.013 (n=5) | 0.838±0.010 (n=5) | 0.833±0.019 (n=5) |
| aifb | 0.994±0.001 (n=5) | 0.995±0.001 (n=5) | 0.993±0.002 (n=5) | 0.992±0.002 (n=5) | – |

## Level 2 — pair-vicinity EPD (decoder concat, TLC-GNN 식)

| 데이터셋 | base | +noise | +mix(CN) | +pair-PI | CN 휴리스틱 단독 |
|---|---|---|---|---|---|
| acm | (미완) | 0.896±0.000 (n=1) | (미완) | (미완) | (미완) |
| dblp | (미완) | (미완) | (미완) | (미완) | (미완) |
| aifb | (미완) | 0.972±0.000 (n=1) | (미완) | (미완) | (미완) |

판정: real>mix>noise ⇒ pair-위상 고유 신호 / real≈mix ⇒ CN 수준 정보뿐 / 전부≈base ⇒ LP 에서도 null (AP 는 metrics.json 의 test_ap).

