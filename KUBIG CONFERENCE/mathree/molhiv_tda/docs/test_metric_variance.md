# `pdgnn_tda_3d_elec` test 지표가 "붕 뜬" 이유: balanced test 평가의 함정

## 관찰
동일한 최적 하이퍼파라미터(`lr=1e-4, dropout=0.3, weight_decay=1e-5`)로 `pdgnn_tda_3d_elec`를
재학습했더니 아래처럼 나왔다.

| 구분 | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|
| 이전 sweep (7/7) | 0.7959 | 0.7667 (balanced) |
| 재학습 (7/8) | 0.7858 | **0.7317 (balanced)** |

valid는 −0.01인데 **test만 −0.035**로 유독 크게 떨어졌다. "모델이 망가진 것 아니냐"는
의심이 들 수 있는 수치다. 결론부터: **모델은 나빠지지 않았고, 떨어진 것은 test 평가 방식이
만들어내는 노이즈다.**

## 원인 1 — GPU 부동소수점 비결정성 (valid의 ±0.01)
PDGNN(`PDConv`)은 `scatter(sum/min)` 기반이라 CUDA에서 `atomicAdd`로 합산 순서가
실행마다 달라진다. `torch.manual_seed`는 dropout·셔플 같은 RNG만 고정할 뿐 atomic scatter의
비결정성은 막지 못한다. 50 epoch 누적되면 최종 가중치가 미세하게 달라져 valid가 ±0.01 흔들린다.
MolHIV ROC-AUC는 원래 분산이 커서 이 정도는 정상 범위다.
(코드 diff로 확인: `train_utils.py`의 GPU 최적화 커밋은 학습 수치에 영향을 주는 변경이 없음 —
feature bank를 매 배치 대신 1회 이동, test를 valid 개선 시에만 계산할 뿐 값은 동일.)

## 원인 2 — balanced test 서브샘플링 (test의 ±0.03~ 변동, 진짜 범인)
`pdgnn_tda_3d_elec` 설정은 `balance_test=True`라, test를 **양성:음성 1:1 균형 서브샘플**에서
평가한다(`train_utils._balanced_binary_subset`). 게다가 그 서브샘플을 뽑는 seed가
`test_balance_seed + (best-valid가 나온 epoch)` 이다. 따라서 보고되는 test 값은
(1) 모델 가중치와 (2) **어느 epoch에서 best-valid가 났는지에 따라 바뀌는 랜덤 부분집합**에
동시에 의존한다.

MolHIV test는 양성이 ~3.5%뿐이라 1:1 서브샘플의 크기가 작고, 어떤 부분집합이 뽑히느냐에
따라 balanced ROC-AUC가 크게 출렁인다.

## 증거 — 저장된 체크포인트로 재평가
`results/pdgnn_tda_3d_elec_best.pt`(valid 0.7858짜리 그 모델)를 그대로 두고 test만 다시 쟀다
(`scripts/eval_pdgnn_tda_ckpt.py`, seed 30개):

```
VALID full ROC-AUC             : 0.7858
TEST  full (unbalanced) ROC-AUC : 0.7761   <- 표준 OGB 지표, 안정적
TEST  balanced 1:1 over 30 seeds:
    mean 0.7681 | std 0.0139 | min 0.7309 | max 0.7983
```

해석:
- **표준(full, 불균형) test ROC-AUC는 0.7761** 로, 예전 "0.76"보다 오히려 높다.
  → 모델 품질은 전혀 나빠지지 않았다.
- balanced 1:1 지표는 같은 모델인데도 **subset seed만 바꿔도 0.7309 ~ 0.7983 (폭 0.067,
  std 0.014)** 로 요동친다.
- 학습 로그에 찍힌 **0.7317은 이 balanced 분포의 거의 최솟값(0.7309)** 에 걸린 "운 나쁜 한 방"
  이었을 뿐이다.

## 결론 / 권장
1. **주 지표는 full(불균형) test ROC-AUC (표준 OGB 방식)** 로 본다. 이게 안정적이고 논문 비교와도
   일치한다. → 이 체크포인트 기준 **valid 0.786 / test 0.776**.
2. balanced 1:1 test를 쓰려면 반드시 **여러 seed 평균±표준편차**로 보고한다. 단일 값 하나로
   우열을 논하지 않는다.
3. 모델 비교(예: HAN fusion vs 백본)도 **valid 또는 full-test** 기준으로, 가능하면 **multi-seed
   평균**으로 판단한다.

## 재현 방법
```bash
python -u scripts/eval_pdgnn_tda_ckpt.py \
    --ckpt results/pdgnn_tda_3d_elec_best.pt --device cuda --balance-seeds 30
```
