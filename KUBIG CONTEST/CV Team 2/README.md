# 🧠 Neural Operator 기반 초해상도 재구성  
## Continuous Super-Resolution with SRNO & LNO  

**CV2 Team | 22기 장건희 · 22기 황원준 · 23기 김병현**

---

## 🔎 프로젝트 요약

초해상도를 단순한 픽셀 보간 문제가 아니라  
**함수 → 함수 매핑(Neural Operator)** 문제로 바라보고,

- SRNO 재현  
- Fourier 기반 변형 실험  
- Local Aggregation 구조 비교  
- LNO 구현 및 Loss 설계 실험  

까지 수행한 실험 중심 프로젝트.

---

## 🚀 Core Contributions

- SRNO baseline 재현 및 multi-scale continuous SR 구현  
- Fourier Positional Encoding 기반 spectral bias 완화 실험  
- Local aggregation 방식 변경에 따른 표현력 분석  
- FNO 계열 구조를 Laplace domain으로 확장한 LNO-CoDA 구현  
- Edge-weighted + Spectral Loss 결합 실험  
- SRNO vs LNO inductive bias 비교 분석  

---

# 1️⃣ Introduction

## 1.1 Super Resolution

초해상도(SR)는  
저해상도(LR) 이미지에서 고해상도(HR) 이미지를 복원하는 문제.

### 특징

- LR 하나에 대해 가능한 HR이 여러 개  
- 정보 손실이 존재하는 ill-posed inverse problem  
- 특히 고주파(엣지, 텍스처) 복원이 핵심 난제  
- 단순 interpolation을 넘어 전역적 구조 활용 필요  

---

## 1.2 LIIF

**LIIF (Local Implicit Image Function)**  

이미지를 연속 함수로 모델링  

RGB = f(x, y)

좌표를 입력하면 RGB를 출력하는 구조.

### 장점

- 학습 배율에 종속되지 않음  
- Arbitrary-scale SR 가능  
- 연속적 표현 가능  

### 한계

- Point-wise MLP 구조  
- Global context 부재  
- 고주파 복원 한계  

→ 전역 연산 기반 구조 필요  

---

## 1.3 Neural Operator

### 기존 Neural Network  
벡터 → 벡터 매핑  

### Neural Operator  
함수 → 함수 매핑  

PDE 분야에서 발전한 개념으로,

- 해상도에 덜 의존 (discretization invariance)  
- 적분 기반 전역 연산 수행  
- 연속 함수 관점에서 LR → HR 직접 매핑  

---

# 2️⃣ SRNO (Super Resolution Neural Operator)

## 2.1 Operator 구조 관점

SRNO는 3단 구조를 따른다.

### 1️⃣ Lifting  
LR feature → 고차원 latent space  

### 2️⃣ Integral Operator  
Kernel Integral 기반 전역 연산  

z(x) = Q(K^T V)

- Galerkin-type projection  
- 전역 basis 공유  
- 좌표 전체 동시 고려  

### 3️⃣ Projection  
Latent → RGB  

→ Continuous coordinate query 기반 HR 예측  

---

## 2.2 Architecture

- Encoder: EDSR  
- Model Width: 128  
- Blocks: 8  
- Latent Dimension: 256  
- Kernel Update Step: T = 2  
- Multi-scale training: {2, 3, 4}  

LR(x) → HR(x) operator 직접 학습 구조  

---

# 3️⃣ SRNO Experiments

### 핵심 질문

> Operator 구조에 어떤 inductive bias를 추가하면 달라지는가?

---

## 3.1 Baseline

- Pure SRNO  
- L1 Loss  
- Multi-scale training {2,3,4}  

Operator 구조 자체 성능 확인  

---

## 3.2 SRNO + Fourier Positional Encoding

좌표 입력에 Fourier feature 추가

- L = 5  
- L = 10  

### 목적

- Spectral bias 완화  
- 고주파 함수 근사력 향상  
- L 값에 따른 표현력 vs 안정성 trade-off 분석  

---

## 3.3 SRNO + Local Aggregation

### 기존 SRNO

- 4-corner feature 단순 concat  

### 변형

- LIIF-style local ensemble 적용  
- LR 셀 내부 상대 위치 반영  
- 가까운 corner에 더 큰 weight  

→ 위치 기반 inductive bias 영향 분석  

---

# 4️⃣ LNO (Laplace Neural Operator)

## 4.1 FNO에서 LNO로

FNO는 Fourier transform 기반 spectral operator.

LNO는 이를 확장하여:

- Fourier 대신 Laplace transform 기반 연산 적용  

SRNO → 공간 도메인 integral 기반  
LNO → spectral domain 기반  

---

## 4.2 LNO-CoDA

- Model Width: 64  
- Blocks: 4  
- Channel Mixer (CoDA)  
- Bicubic + Residual Learning  

입력 Bicubic 결과에 residual을 더하는 구조  
→ 모델이 고주파 복원에 집중  

---

## 4.3 LNO-CoDA-Big

확장 모델

- Channel mixing 강화  
- Feature capacity 증가  
- Depth 확장  

Spectral operator 확장 가능성 탐색  

---

## 4.4 Loss Design (LNO Series)

### Pixel Loss (Edge-weighted)

- Sobel edge map 추출  
- 엣지 영역 가중치 증가  
- 경계 복원 강화  

### Spectral Loss

- FFT 공간 L1  
- 고주파 성분 직접 제어  

### Big 모델

- Pixel + Spectral 가중 합  
- Edge-weight 비율 조정  

→ 공간 + 주파수 동시 최적화  

---

# 5️⃣ Dataset

## DIV2K

- Train: 800  
- Valid: 100  
- HR Patch: 192 × 192  

### LR 생성

- Bicubic downsampling  
- Scale: ×2 / ×3 / ×4  

---

# 6️⃣ Training Setup

## SRNO

- n_samples: 4096  
- Loss: L1  
- Adam (lr = 4e-5)  
- Warmup: 20  
- Batch: 4  
- Gradient clipping: 1.0  

---

## LNO

- Loss: L1 + Spectral (+ Edge-weight)  
- AdamW (lr = 1e-5)  
- Warmup + CosineAnnealing  
- Batch: 32  
- Epochs: 2000  

---

# 7️⃣ Evaluation

## Metric

- PSNR (Y-channel)  
- DIV2K validation 100장 평균  

## 비교 모델

- Bicubic  
- SRNO Baseline  
- Fourier L=5  
- Fourier L=10  
- Local Aggregation  
- LNO / LNO-CoDA / LNO-CoDA-Big  

---

# 8️⃣ Results & Analysis

## 8.1 Fourier Experiment

- 평균 PSNR 변화는 크지 않음  
- 고주파-rich 패치에서 L=10 우세  
- smooth 영역에서는 L=5 안정적  
- scale이 커질수록 L=10 유리  

→ 고주파 근사 측면에서 부분적 이득 확인  

---

## 8.2 Local Aggregation

- 모든 scale에서 baseline 대비 PSNR 감소  
- 고정 가중 결합 → 표현력 제한  
- SR에서는 dynamic feature 조합 중요  

---

## 8.3 LNO 계열 관찰

- 일부 입력 복사 경향  
- 전역 spectral 연산이 국소 디테일 복원에 불리 가능  
- 공간/주파수 균형 중요  
- Loss 설계에 따라 학습 안정성 크게 변동  

→ PDE 기반 operator 가정과 이미지 SR 간 구조 차이 존재  

---

# 9️⃣ Conclusion

- Continuous SR 관점에서 SRNO 구조 구현 및 분석  
- Fourier PE로 고주파 근사 개선 가능성 확인  
- Local aggregation은 단순 도입 시 성능 저하  
- LNO는 Laplace 기반 spectral 확장 모델  
- Operator 기반 SR의 가능성과 한계 동시 확인  

---

# 🔟 Limitations

- DIV2K 단일 데이터셋  
- PSNR 중심 평가  
- Continuous 좌표 샘플링 variance  
- PDE 기반 가정과 이미지 SR 구조 차이  
- 다양한 Operator 구조 범용 검증 부족  

---

# 🔮 Future Works

- Adaptive Fourier Gating  
- Attention 기반 Dynamic Local Aggregation  
- Hybrid Spatial–Spectral Operator  
- Multi-scale Stability-aware Spectral Loss  
- LPIPS / Perceptual 중심 평가 확장  
