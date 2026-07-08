# PDGNN(frozen) + typed HAN + Multifiltration fine-tune — 결과 비교

## 실험 개요

**목표**: 사전학습된 `pdgnn_tda_3d_elec` 백본을 동결(freeze)하고, 원소별 node type을 갖는
heterogeneous HAN 브랜치 + multifiltration TDA(A/B) 렌즈를 head에 추가한 fine-tune 모델의
valid/test 성능을 백본 및 이전 homogeneous HAN 버전과 비교한다.

| 항목 | 값 |
|---|---|
| config | `pdgnn_han_finetune_3d_elec_multifilt` |
| backbone ckpt | `results/pdgnn_tda_3d_elec_best.pt` |
| HAN | typed node type (13 elements) + bond relation (5 types), node-level gated fusion |
| TDA | BondTDA(C) 50d + 3D TDA 75d + Multifiltration A+B 100d |
| edge | electrostatic 2d (`edge_phys`) |
| train | balanced_train (1:1 pos/neg oversampling) |
| sweep | 4 combo × 3 seeds = 12 runs, ranked by mean valid ROC-AUC |
| best retrain | lr=3e-4, head dropout=0.3, han_hidden=128, 30 epochs, seed 0 |
| eval | valid full + test full (OGB) + test balanced 30-seed (`eval_pdgnn_han_finetune_ckpt.py --multifilt`) |

## 최종 성능 비교 (test_full 기준)

| 모델 | VALID full | TEST full | TEST balanced (30-seed) |
|---|---:|---:|---|
| **Backbone** `pdgnn_tda_3d_elec` | 0.7858 | **0.7761** | 0.7681 ± 0.0139 |
| Fine-tune homogeneous HAN (no multifilt) | 0.8293 | 0.7601 | 0.7561 ± 0.0162 |
| **Fine-tune typed HAN + multifilt (본 실험)** | **0.8488** | 0.7756 | **0.7741 ± 0.0154** |

차이 (multifilt fine-tune − backbone):

| 지표 | Δ |
|---|---:|
| VALID full | **+0.0630** |
| TEST full | **−0.0005** |
| TEST balanced mean | +0.0060 |

## 해석

1. **VALID는 크게 상승** (+0.063) — HAN + multifiltration + balanced train이 검증셋에는 잘 맞음.
2. **TEST full은 백본과 사실상 동일** (−0.0005) — OGB 표준 지표 기준 generalization 이득 없음.
3. **TEST balanced**는 +0.006이나 std(±0.015) 범위 내 — [`test_metric_variance.md`](test_metric_variance.md)에서
   설명한 balanced 지표 노이즈를 고려하면 유의미한 개선으로 보기 어렵다.
4. homogeneous HAN( multifilt 없음) 대비 multifilt + typed node type 추가 시 valid +0.0195,
   test_full +0.0155로 소폭 개선되었으나 여전히 백본 test_full에는 미달하지 않음.

**결론**: 현재 아키텍처/하이퍼파라미터에서는 valid 과적합 패턴이 두드러지며,
**test_full 기준 백본 대비 실질적 이점은 없다.**

## Phase 2 스윕 순위 (3-seed mean valid, test=balanced during sweep)

| rank | dropout | han_hidden | valid mean±std | test_bal mean±std |
|---:|---:|---:|---:|---:|
| 1 | 0.3 | 128 | **0.8410 ± 0.0070** | 0.7559 ± 0.0151 |
| 2 | 0.5 | 128 | 0.8410 ± 0.0055 | 0.7644 ± 0.0064 |
| 3 | 0.3 | 256 | 0.8374 ± 0.0046 | 0.7637 ± 0.0151 |
| 4 | 0.5 | 256 | 0.8339 ± 0.0099 | 0.7799 ± 0.0104 |

1위 combo(`do0.3/hh128`)로 `--save-ckpt` 재학습 후 multi-seed eval 수행.

## 재학습 best checkpoint eval (공식)

```
ckpt: results/pdgnn_han_finetune_3d_elec_multifilt_best.pt
VALID full ROC-AUC             : 0.8488
TEST  full (unbalanced) ROC-AUC : 0.7756
TEST  balanced 1:1 over 30 seeds:
    mean 0.7741 | std 0.0154 | min 0.7425 | max 0.8030
```

Trainable params: 1,011,018 | Frozen (backbone): 2,566,202

## 아키텍처 요약

```
[Frozen PDGNN+elec] → node_x [N,600]
[Typed HAN]         → gate·h_han [N,128]   (13 atom node types, 5 bond relations)
        ↓ concat + global_add_pool → [B,728]
+ BondTDA(50) + 3DTDA(75) + Multifiltration A+B(100)
        ↓
New head → HIV logit
```

## 재현

```bash
# sweep (already done)
python -u scripts/sweep_pdgnn_han_finetune.py \
  --config pdgnn_han_finetune_3d_elec_multifilt \
  --backbone-ckpt results/pdgnn_tda_3d_elec_best.pt \
  --balanced-trains 1 --lrs 3e-4 --dropouts 0.3,0.5 \
  --han-hiddens 128,256 --seeds 0,1,2 --epochs 30 --device cuda \
  --max-parallel 5

# best retrain + eval
python -u train/train_pdgnn_han_finetune.py \
  --config pdgnn_han_finetune_3d_elec_multifilt \
  --backbone-ckpt results/pdgnn_tda_3d_elec_best.pt \
  --lr 3e-4 --dropout 0.3 --han-hidden 128 --balanced-train \
  --epochs 30 --save-ckpt results/pdgnn_han_finetune_3d_elec_multifilt_best.pt \
  --out results/pdgnn_han_finetune_3d_elec_multifilt_best.json

python -u scripts/eval_pdgnn_han_finetune_ckpt.py \
  --ckpt results/pdgnn_han_finetune_3d_elec_multifilt_best.pt --multifilt \
  --han-hidden 128 --dropout 0.3 --balance-seeds 30
```

## 관련 결과 파일

- `results/pdgnn_han_finetune_3d_elec_multifilt_best.json`
- `results/sweep/pdgnn_han_finetune_3d_elec_multifilt_hantune/_summary.json`
- 로그: `logs/sweep_hantune_multifilt.log`, `logs/finetune_pipeline_multifilt.log`
