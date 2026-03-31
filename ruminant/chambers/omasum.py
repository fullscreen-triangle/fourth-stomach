"""
Chamber 3: Omasum (Graph Completion Attention)

Identifies tokens with high epistemic uncertainty and completes their
representations using the spectral correlation graph from Chamber 2.
Information flows ONLY from confident to uncertain tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmasumLayer(nn.Module):
    """Graph completion attention: fills representation gaps.

    Unlike standard attention where all tokens influence all others,
    the Omasum directs information from low-uncertainty to high-uncertainty
    tokens only. This prevents contamination and ensures monotonic gap reduction.
    """

    def __init__(self, d_model: int, correlation_threshold: float = 0.3,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.threshold = correlation_threshold
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Learnable gating network
        self.gate = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Sigmoid(),
        )

    def _token_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token uncertainty via representation entropy."""
        p = F.softmax(x.abs(), dim=-1)
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # (B, N)
        return entropy

    def forward(self, x: torch.Tensor, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            S: (batch, seq_len, seq_len) — spectral correlation matrix from Chamber 2
        Returns:
            out: (batch, seq_len, d_model) — gap-completed representation
            unc_before: (batch, seq_len) — uncertainty before completion
            unc_after:  (batch, seq_len) — uncertainty after completion
        """
        B, N, D = x.shape
        residual = x

        unc_before = self._token_uncertainty(x)  # (B, N)

        # Completion graph: flow from low uncertainty to high
        # G_ij > 0 iff uncertainty[j] < uncertainty[i] AND S[i,j] > threshold
        unc_i = unc_before.unsqueeze(-1)  # (B, N, 1)
        unc_j = unc_before.unsqueeze(-2)  # (B, 1, N)
        mask = (unc_j < unc_i) & (S > self.threshold)
        G = S * mask.float()  # (B, N, N)

        # Normalise completion graph
        G_sum = G.sum(dim=-1, keepdim=True) + 1e-8
        G_norm = G / G_sum

        # Gating: lambda_i controls how much completion to apply per token
        unc_relative = (unc_before - unc_before.mean(dim=-1, keepdim=True)) / (unc_before.std(dim=-1, keepdim=True) + 1e-8)
        lam = self.gate(unc_relative.unsqueeze(-1))  # (B, N, D)

        # Completion: weighted average of confident neighbours
        completed = torch.bmm(G_norm, x)  # (B, N, D)
        out = self.norm(residual + self.dropout(lam * (completed - residual)))

        unc_after = self._token_uncertainty(out)
        return out, unc_before, unc_after
