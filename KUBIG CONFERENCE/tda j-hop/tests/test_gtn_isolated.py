"""회귀 테스트: 모든 base relation 에서 고립된 노드가 있어도 GTN 학습이 NaN 이 안 남.

버그(수정 전): 고립 노드는 inter-layer `_gtn_norm`(대각 제거) 후 all-zero 행 → deg=0 →
`torch.where(deg>0, 1/deg, 0)` 의 backward 에서 0×inf=NaN → 첫 backward 에 GTConv 가중치
전체 NaN 고착 (imdb 10/10 · freebase 9/10 run 재현 원인).
"""
import torch
import torch.nn.functional as F

from tda.models.gtn import GTN


def _toy_A_with_isolated(n=20, n_iso=3, seed=0):
    """관계 2개 + identity. 마지막 n_iso 개 노드는 모든 관계에서 고립."""
    g = torch.Generator().manual_seed(seed)
    rels = []
    for _ in range(2):
        a = (torch.rand(n, n, generator=g) > 0.6).float()
        a = ((a + a.t()) > 0).float()
        a.fill_diagonal_(0.0)
        a[-n_iso:, :] = 0.0
        a[:, -n_iso:] = 0.0          # 고립 노드
        rels.append(a)
    rels.append(torch.eye(n))
    return torch.stack(rels, dim=0)


def test_gtn_training_finite_with_isolated_nodes():
    torch.manual_seed(0)
    n = 20
    A = _toy_A_with_isolated(n)
    x = torch.randn(n, 8)
    y = torch.randint(0, 3, (n,))
    model = GTN(num_relations=A.size(0), num_channels=2, in_dim=8, hidden_dim=16,
                num_classes=3, num_layers=2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(10):                     # 버그 시 첫 backward 에 이미 NaN
        opt.zero_grad()
        logits, H = model(A, x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), "loss became NaN/inf"
    for p in model.parameters():
        assert torch.isfinite(p).all(), "weights became NaN/inf"
    for layer in model.channel_attentions():
        for w in layer:
            assert torch.isfinite(w).all(), "GTN attention NaN"
