"""
Chamber 4: Abomasum (Refinement Attention)

Confidence-weighted self-attention: high-confidence tokens have more
influence on the output. Implements backward trajectory refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbomasumLayer(nn.Module):
    """Refinement attention: the 'true stomach'.

    Standard self-attention modified so that each key's contribution
    is weighted by its confidence (inverse uncertainty). High-confidence
    tokens dominate the output; uncertain tokens are suppressed.
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

    def forward(self, x: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            uncertainty: (batch, seq_len) — per-token uncertainty from Chamber 3
        Returns:
            (batch, seq_len, d_model) — refined representation
        """
        B, N, D = x.shape
        residual = x

        # Confidence = inverse of normalised uncertainty
        confidence = 1.0 - uncertainty / (uncertainty.max(dim=-1, keepdim=True).values + 1e-8)  # (B, N)

        qkv = self.W_qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Confidence weighting: multiply attention scores by key confidence
        # confidence shape: (B, N) -> (B, 1, 1, N) for broadcasting
        conf_weights = confidence.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
        scores = scores + torch.log(conf_weights + 1e-8)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.W_out(out)

        return self.norm(residual + self.dropout(out))
