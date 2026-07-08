"""Bond-type learnable-alpha weighted fusion of PDGNN output and MW."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BondTypeWeightedFusion(nn.Module):
    """
    Fuse g_pdg and mw_emb with bond-composition-modulated attention.

    score_k = a^T · h_k  (no additive bond bias)
    α_g = bond_frac @ alpha   (learnable alpha ∈ R^{5×2})
    attn = softmax([score_pdg, score_mw] * α_g)
    fused = attn_0 · g_pdg + attn_1 · mw_emb
    """

    def __init__(self, dim: int, num_bond_types: int = 5):
        super().__init__()
        self.dim = dim
        self.mw_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.score_proj = nn.Linear(dim, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(num_bond_types, 2))

    def forward(
        self,
        g_pdg: torch.Tensor,
        mw: torch.Tensor,
        bond_frac: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mw_emb = self.mw_proj(mw)
        score_pdg = self.score_proj(g_pdg)
        score_mw = self.score_proj(mw_emb)
        scores = torch.cat([score_pdg, score_mw], dim=-1)

        bond_scale = bond_frac @ self.alpha
        scores = scores * bond_scale
        attn = F.softmax(scores, dim=-1)

        fused = attn[:, 0:1] * g_pdg + attn[:, 1:2] * mw_emb
        return fused, attn
