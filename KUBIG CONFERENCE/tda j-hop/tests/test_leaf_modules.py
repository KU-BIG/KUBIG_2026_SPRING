"""Leaf 모듈(utils, persistence_image, hks, pdgnn) 단위 테스트."""
import numpy as np
import torch

from tda.utils import macro_f1, accuracy, set_seed
from tda.topology.persistence_image import PersistenceImage
from tda.topology.hks import compute_hks
from tda.models.pdgnn import PDGNN, bipartite_loss


def test_macro_f1_perfect():
    logits = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    labels = torch.tensor([0, 1, 2])
    assert abs(macro_f1(logits, labels) - 1.0) < 1e-9
    assert abs(accuracy(logits, labels) - 1.0) < 1e-9


def test_macro_f1_handles_missing_class():
    # 한 클래스만 맞히면 macro-f1 < accuracy 일 수 있음 — 0..1 범위만 확인
    logits = torch.tensor([[5.0, 0], [5.0, 0], [5.0, 0]])
    labels = torch.tensor([0, 0, 1])
    f1 = macro_f1(logits, labels)
    assert 0.0 <= f1 <= 1.0


def test_persistence_image_shape_and_empty():
    pi = PersistenceImage(resolution=5)
    assert pi.dim == 25
    assert pi.transform(np.zeros((0, 2))).shape == (25,)
    assert np.allclose(pi.transform(np.zeros((0, 2))), 0.0)


def test_persistence_image_nonneg_and_weight():
    pi = PersistenceImage(resolution=5, pers_range=(0.0, 6.0))
    # death<=birth 인 점은 무시되어야 함
    pairs = np.array([[0.0, 3.0], [1.0, 1.0], [0.0, -1.0]])
    vec = pi.transform(pairs)
    assert vec.shape == (25,)
    assert np.all(vec >= 0.0)
    assert vec.sum() > 0.0  # 유효 점 (0,3) 이 기여


def test_compute_hks_shape_and_znorm():
    # 작은 path 그래프 0-1-2-3
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    hks = compute_hks(ei, num_nodes=4, K=3, device=torch.device("cpu"))
    assert hks.shape == (4, 3)
    # 각 스케일 z-정규화: 평균≈0
    assert np.allclose(hks.mean(axis=0), 0.0, atol=1e-6)


def test_pdgnn_forward_shapes():
    set_seed(0)
    model = PDGNN(hidden_dim=8, num_layers=2)
    filt = torch.randn(5, 1)
    ei = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1, 2, 3]], dtype=torch.long)
    pd = model(filt, ei)
    assert pd.shape == (ei.size(1), 2)


def test_bipartite_loss_basic():
    pred = torch.tensor([[0.0, 1.0], [0.0, 2.0]], requires_grad=True)
    gt = torch.tensor([[0.0, 1.0]])
    loss = bipartite_loss(pred, gt)
    assert loss.item() >= 0.0
    loss.backward()
    assert pred.grad is not None
    # 빈 입력 안전
    assert bipartite_loss(torch.zeros((0, 2)), gt).item() == 0.0
