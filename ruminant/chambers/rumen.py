"""
Chamber 1: Rumen (Circulation Attention)

Dense token mixing with full-context self-attention.
Analogous to the biological rumen: raw ingestion without selectivity.
Enforces KCL (token conservation) via softmax normalisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RumenLayer(nn.Module):
    """Standard multi-head self-attention with residual and layer norm.

    The Rumen ingests everything. Its role is to mix: every token attends
    to every other token, establishing the initial circulation pattern.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) — circulation-mixed representation
        """
        B, N, D = x.shape
        residual = x

        qkv = self.W_qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, H, d_head)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, H, N, d_head)

        # Scaled dot-product attention (KCL: softmax ensures sum = 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, N, d_head)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.W_out(out)

        return self.norm(residual + self.dropout(out))
