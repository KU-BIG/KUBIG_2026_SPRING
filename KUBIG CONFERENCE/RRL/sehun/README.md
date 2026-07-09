# RRL (Robust RL) : Adaptive Method — adversarial agent 탐지 후 policy 전환하기

3인 협동 Overcooked 환경(`dec_3_chefs_secret_heaven`)에서, 팀에 이기적으로 협동을 방해하는 에이전트가 섞여 있어도 잘 협동하게 만드는 게 목표. 베이스인 N-XPlay / SPN_1ADV 위에, 팀에서 제안하는 detector 기반 adaptive 방식을 구현하고 돌려본 코드와 결과를 모음.

## baseline Policies

- **SP** — 정상 팀원끼리만 self-play로 학습. 정상 상황엔 강하지만 적대자가 끼면 무너짐.
- **ADV** — 개인 보상만 챙기며 팀을 방해하는 적대적 에이전트(selfisher).
- **SPN_1ADV** — 학습할 때 절반은 정상 팀, 절반은 적대자가 낀 팀으로 섞어서 돌린 정책. 두 상황 모두 어느 정도 버팀

## 아이디어

에피소드 앞부분(400스텝 중 100스텝)을 일단 SP로 플레이하면서, 그동안 쌓인 관측만 보고 RNN(GRU) detector가 "지금 팀에 적대자가 있나 없나"를 맞힌다. 판정 결과에 따라 남은 구간의 정책을 바꾼다 — 없다고 보면 그대로 SP, 있다고 보면 견고한 SPN_1ADV로 갈아탐.

상황별로 특화된 정책을 골라 쓰면 늘 SPN_1ADV만 쓰는 것보다 나을까? 를 확인해보려는 게 핵심

## 폴더 구성

```
sehun/
├── adaptive/
│   ├── common.py          환경·에이전트·롤아웃 공용 헬퍼
│   ├── gen_data.py        detector 학습용 데이터 생성 (정상/적대 에피소드 롤아웃)
│   ├── train_detector.py  GRU detector 학습 (env 없이 numpy+torch만)
│   ├── eval_adaptive.py   SP / SPN_1ADV / Adaptive 비교 평가
│   └── run_gen.sh, run_eval.sh   tmux 실행용 런처
└── results/
    ├── detector.pt        학습된 detector 가중치 + 정규화 통계
    └── eval_results.json  평가 원시 결과와 요약
```

## 실행

[multiHRI(N-XPlay)](https://github.com/HIRO-group/multiHRI) 프레임워크와 학습된 체크포인트(`agent_models/SecretHeaven_N3/...`, 용량이 커서 여기엔 올리지 못했습니다)가 갖춰진 환경에서 돌림

```bash
source <venv>/bin/activate
export MULTIHRI_ROOT=/path/to/multiHRI        # 기본값 /workspace/rl_project/multiHRI
export PYTHONPATH=$MULTIHRI_ROOT SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

python adaptive/gen_data.py --per_class 250   # 데이터 생성 (앞 100스텝 관측)
python adaptive/train_detector.py             # detector 학습
python adaptive/eval_adaptive.py --episodes 40  # 세 방법 비교
```

데이터 생성과 평가는 시간이 좀 걸려서 `run_gen.sh` / `run_eval.sh`를 tmux로 돌리면 연결이 끊겨도 계속 진행

## 결과

detector는 앞 100스텝만 보고도 적대자 유무를 거의 정확히 맞힘 (validation 97.0%, 평가 중 실측 78/80 = 97.5%).

정책 비교 (조건별 40 에피소드, 값은 팀 보상 합계. 강건성 = 적대적 ÷ 정상):

| 방법 | 전체 평균 | 정상 | 적대적 | 강건성 |
|:--|--:|--:|--:|--:|
| SP | 273.00 | 393.50 | 152.50 | 38.8% |
| SPN_1ADV | 474.50 | 507.00 | 442.00 | 87.2% |
| Adaptive | 365.75 | 390.00 | 341.50 | 87.6% |

## 정리

Adaptive는 적대자 상황 강건성(87.6%)은 SPN_1ADV(87.2%)에 거의 붙었지만, 전체 평균에서는 넘지 못함. 이유는 둘인데, 하나는 정상 상황에서 Adaptive가 SP를 쓰는데, 이번 실험에선 SPN_1ADV가 정상에서도 더 잘해서(507 vs 390) "정상=SP가 유리"라는 전제가 깨진 점이고 다른 하나는 적대 상황에서 앞 100스텝을 SP로 흘려보내는 동안 점수를 손해 본 것(341.5 vs 442, detection delay).

결국 상황을 나눠 특화 정책을 고르는 것보다, 처음부터 두 상황을 섞어 학습한 단일 정책(SPN_1ADV)이 더 나아 원하던 결과와는 조금 다르게 되었음. 발표에서 이야기하게 될 결론(OOD와 detection delay 때문에 adaptive가 SPN_1ADV에 못 미친다)과 동일한 결과.

## 향후 과제

SPN_1ADV를 넘어서려면 detection delay를 줄이거나(더 이르게, 혹은 계속 재판정), 두 정책을 섞어 쓰거나, detector 신호를 정책 입력에 직접 넣는 end-to-end 방식을 시도해봐야할 것으로 생각됨.
