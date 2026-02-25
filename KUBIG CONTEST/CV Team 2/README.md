🧠 Neural Operator 기반 초해상도 재구성
Continuous Super-Resolution with SRNO & LNO
CV2 Team | 22기 장건희 · 22기 황원준 · 23기 김병현

📌 Overview
본 프로젝트는 Neural Operator 기반 초해상도(Super-Resolution) 모델을 구현하고,
구조적 변형을 통해 성능 및 표현력 변화를 분석한 연구입니다.
기존 CNN 기반 초해상도는 고정된 해상도 매핑에 초점을 둡니다.
본 연구는 이미지를 연속 함수(continuous function) 로 재정의하고,
LR 함수 → HR 함수로 매핑하는 Operator 학습
이라는 관점에서 초해상도 문제를 접근합니다.
이를 위해:
SRNO (Super Resolution Neural Operator)


Fourier Positional Encoding


Local Aggregation


LNO (Laplace Neural Operator) + CoDA


를 구현하고 비교 분석했습니다.

1️⃣ Background
🔹 Super Resolution
초해상도(SR)는 저해상도 이미지로부터 고해상도 이미지를 복원하는 문제입니다.
하지만,
하나의 LR 이미지는 여러 HR 이미지로부터 생성 가능


즉, 다대일 → 역문제


고주파 정보가 손실된 상태에서 복원해야 함


따라서 본질적으로 ill-posed problem 입니다.

🔹 LIIF (Local Implicit Image Function)
LIIF는 이미지를 좌표 기반 연속 함수로 정의합니다.
RGB = f(x, y)
장점:
학습 배율에 묶이지 않음


arbitrary-scale SR 가능


연속적 표현


한계:
Point-wise MLP


Global context 부재


High-frequency 복원 한계


→ 이미지 전체를 함수 관점에서 다루는 구조 필요

🔹 Neural Operator
기존 Neural Network: 벡터 → 벡터
Neural Operator: 함수 → 함수
PDE 문제에서 발전


Discretization Invariance


적분 연산 기반 Global Context 처리


이를 이미지 초해상도에 적용한 구조가 SRNO 입니다.

2️⃣ Models

🧩 SRNO (Super Resolution Neural Operator)
Core Idea
이미지를 discrete pixel 집합이 아닌 연속 함수로 보고,
LR(x) → HR(x)
를 직접 학습하는 operator 구조.
Architecture
Encoder: EDSR


Model Width: 128


Blocks: 8


Latent dim: 256


Kernel update step T = 2


Continuous coordinate query


Kernel Integral 형태:
z(x) = Q(K^T V)
O(n²) attention 대비 계산 효율적


전역 basis 공유



🔹 SRNO Variants
1️⃣ Baseline
Pure SRNO


L1 Loss


Multi-scale training {2,3,4}



2️⃣ SRNO + Fourier Positional Encoding
좌표에 Fourier feature 추가:
L=5


L=10


목적:
Spectral bias 완화


고주파 함수 근사 능력 향상



3️⃣ SRNO + Local Aggregation
기존 SRNO:
4-corner feature 단순 concat


변형:
LIIF-style local ensemble


상대 위치 기반 가중 결합


inductive bias 추가



🧩 LNO (Laplace Neural Operator)
SRNO와 다른 spectral 접근.
LNO-CoDA 구조
Model Width: 64


Blocks: 4


Channel Mixer (CoDA)


Residual learning (Bicubic + Residual)


Loss:
Pixel Loss (Edge-weighted)


Spectral Loss (FFT domain L1)


목표:
고주파 복원에 집중


채널 간 상호작용 강화



3️⃣ Dataset
DIV2K
Train: 800 HR images


Valid: 100 HR images


HR patch size: 192 × 192


LR 생성:
Bicubic downsampling


Scale: ×2 / ×3 / ×4



4️⃣ Training Setup
SRNO
HR patch: 192×192


Multi-scale: {2,3,4}


n_samples: 4096


Loss: L1


Optimizer: Adam (4e-5)


Warmup: 20


Batch size: 4


Gradient clipping: 1.0



LNO
HR patch: 192×192


Train scale: {2, 3, 4}


Loss: L1 + Spectral


Optimizer: AdamW (1e-4)


Scheduler: Warmup + CosineAnnealing


Batch size: 32


Epochs: 2000



5️⃣ Evaluation
Metric
PSNR (dB), Y-channel 기준


100 validation 이미지 평균


비교 대상:
Bicubic


Baseline


Fourier L=5


Fourier L=10


LocalAgg



6️⃣ Results
🔹 Fourier Experiment
평균 PSNR 변화는 크지 않음


고주파-rich patch에서 L=10 우세


smooth patch에서 L=5 우세


scale 커질수록 L=10 상대적 우세


→ L 동적 선택 Gating 구조 가능성 제시

🔹 Local Aggregation
모든 scale에서 Baseline 대비 PSNR 감소


고정 가중치 결합이 표현력 제한


SR 문제에서 dynamic feature 조합 중요성 확인



🔹 LNO Observation
일부 경우 입력을 그대로 출력하는 경향


전역 spectral 연산이 국소 정보 무시 가능성


PDE 기반 operator와 이미지 복원 간 구조적 차이 존재



7️⃣ Key Insights
Neural Operator는 continuous SR을 자연스럽게 수행 가능


Fourier PE는 고주파 근사에 도움


Local ensemble은 단순 도입 시 성능 저하


SR에서는 dynamic basis 조합이 핵심


Operator 구조는 CNN과 다른 inductive bias를 가짐



8️⃣ Limitations
DIV2K 단일 데이터셋


PSNR 중심 평가


다양한 operator 구조에 대한 일반화 부족


Continuous sampling 구조로 인한 high variance


PDE 기반 구조와 이미지 복원의 본질적 차이



9️⃣ Future Work
Gated Fourier L selection


Dynamic local aggregation


FLOPs 비교 분석


Perceptual metric (LPIPS, SSIM)


다른 Neural Operator 확장


Larger patch training



🔎 Project Significance
본 프로젝트는 “Neural Operator를 이미지 초해상도에 적용한 SRNO, FNO 기반 구조적 변형을 통해 표현력 및 안정성 변화를 실험적으로 분석한 연구”입니다.
Continuous SR 관점에서 operator 기반 접근의 가능성과 한계를 동시에 탐구했다는 점에서 의미를 가집니다.
