# 실험 ① — 주입(concat vs gate) × 내용(real/noise/mix) factorial (RGCN, NC)

질문: (f)<(d) 의 위상 유해가 **concat 주입 방식** 탓인가? 위상 = manual 채널·topo_seed 고정(조건 간 동일 특징 — GTN/per-seed 아님). 모든 조건 동일 GatedRGCN 구현. test Macro-F1, mean±std.

| 데이터셋 | base | concat+real | concat+noise | concat+mix | gate+real | gate+noise | gate+mix |
|---|---|---|---|---|---|---|---|
| acm | 0.930±0.009 (n=10) | 0.905±0.000 (n=10) | 0.926±0.007 (n=10) | 0.932±0.006 (n=10) | 0.916±0.000 (n=10) | 0.924±0.011 (n=10) | 0.933±0.005 (n=10) |
| dblp | 0.937±0.004 (n=10) | 0.930±0.000 (n=10) | 0.933±0.006 (n=10) | 0.936±0.003 (n=10) | 0.939±0.000 (n=10) | 0.923±0.007 (n=10) | 0.938±0.003 (n=10) |
| aifb | 0.751±0.013 (n=10) | 0.760±0.000 (n=10) | 0.685±0.075 (n=10) | 0.676±0.045 (n=10) | 0.732±0.000 (n=10) | 0.684±0.065 (n=10) | 0.728±0.017 (n=10) |

판정 가이드: cat_real<base 이면서 gate_real≥base ⇒ 하락은 주입(concat) 탓. gate_real>gate_mix ⇒ 게이팅에선 per-node 정렬 기여 존재.

