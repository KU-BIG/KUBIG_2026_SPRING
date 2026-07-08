# TLC-GNN / PDGNN Colab 실행 가이드

## 사전 준비 (파일 2개 필요)
레포에 포함된 패치 파일:
- `dgformat.py`
- `riccidist2dgm.py`

---

## 실행 순서

### 셀 1 — 레포 클론
```python
!git clone https://github.com/pkuyzy/TLC-GNN.git
%cd TLC-GNN
```

### 셀 2 — PyG 및 의존성 설치
```python
import torch
v = torch.__version__.split('+')[0]
cuda = 'cu' + torch.version.cuda.replace('.', '')

!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-{v}+{cuda}.html -q
!pip install torch-geometric gudhi networkx scikit-learn GraphRicciCurvature -q
```

### 셀 3 — 패치 파일 적용
> `dionysus` 라이브러리가 Python 3.11에서 빌드 불가능하므로
> `gudhi`로 대체한 패치 파일 2개를 덮어씁니다.

Colab 왼쪽 📁 파일탭에서 아래 경로로 드래그앤드롭:
- `dgformat.py` → `/content/TLC-GNN/sg2dgm/dgformat.py`
- `riccidist2dgm.py` → `/content/TLC-GNN/sg2dgm/riccidist2dgm.py`

또는 셀에서:
```python
# 구글 드라이브에 패치 파일을 올려뒀을 경우
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('/content/drive/MyDrive/patch/dgformat.py',
            '/content/TLC-GNN/sg2dgm/dgformat.py')
shutil.copy('/content/drive/MyDrive/patch/riccidist2dgm.py',
            '/content/TLC-GNN/sg2dgm/riccidist2dgm.py')
print("패치 완료")
```

### 셀 4 — sg2dgm 빌드
```python
%cd /content/TLC-GNN/sg2dgm
!python setup_PI.py build_ext --inplace
%cd /content/TLC-GNN
```
> 빌드 실패 시:
> ```python
> !cp /content/TLC-GNN/sg2dgm/persistenceImager.pyx \
>     /content/TLC-GNN/sg2dgm/persistenceImager.py
> ```

### 셀 5 — 실행할 데이터셋 선택
```python
# pipelines.py 기본값은 ["Photo", "PubMed", "Computers"] → 수 시간 소요
# 빠른 테스트용으로 Cora만 실행
!sed -i 's/d_names = \["Photo", "PubMed", "Computers"\]/d_names = ["Cora"]/' \
    /content/TLC-GNN/pipelines.py

!grep "d_names" /content/TLC-GNN/pipelines.py  # 확인
```

### 셀 6 — 실행
```python
!python /content/TLC-GNN/pipelines.py
```

---

## 데이터셋별 예상 소요 시간

| 데이터셋 | 소요 시간 | 비고 |
|---|---|---|
| Cora | 5~10분 | 테스트용 권장 |
| Citeseer | 10~20분 | |
| PubMed | 30분~1시간 | |
| Photo | 1~3시간 | |
| Computers | 3~6시간 | |
| **전체** | **5~10시간** | GPU 유료 권장 |

> 전체 실행 시 Colab Pro (월 $10) 또는 유료 GPU 사용 권장  
> 무료 티어는 세션이 끊길 수 있음 (브라우저 탭은 열어둘 것)

---

## 핵심 변경 사항 요약 (왜 패치했나)

| 문제 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: dionysus` | Python 3.11에서 빌드 불가 | `gudhi`로 대체 |
| `np.float()` 오류 | numpy 최신버전에서 제거됨 | `float()`로 교체 |
| Python 3.7 요구 | 레포 기본 요구사항 | 3.11에서도 동작 확인 |
