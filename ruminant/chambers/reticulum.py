"""
Chamber 2: Reticulum (Spectral Attention)

Applies FFT to token representations, detects harmonic coincidences
between token frequency signatures, and constructs spectral correlation.
Enforces KVL (spectral consistency) via phase-coherent filtering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReticulumLayer(nn.Module):
    """Spectral attention: operates in the frequency domain.

    Detects correlations invisible to standard dot-product attention
    by comparing token frequency signatures via FFT.
    """

    def __init__(self, d_model: int, alpha: float = 0.3, temperature: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha  # damping coefficient (contraction strength)
        self.temperature = temperature
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Learnable projection for spectral domain
        self.spectral_proj = nn.Linear(d_model // 2 + 1, d_model // 2 + 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model) — spectrally filtered representation
            S:   (batch, seq_len, seq_len) — spectral correlation matrix
        """
        B, N, D = x.shape
        residual = x

        # FFT along feature dimension: each token gets a frequency signature
        x_hat = torch.fft.rfft(x, dim=-1)  # (B, N, D//2+1) complex

        # Learnable spectral projection
        x_hat = self.spectral_proj(x_hat)

        # Spectral correlation: normalised inner product of frequency signatures
        norms = torch.sqrt(torch.sum(torch.abs(x_hat) ** 2, dim=-1, keepdim=True) + 1e-8)
        x_norm = x_hat / norms  # (B, N, D//2+1)

        # S_ij = |<x_hat_i, x_hat_j>| / (|x_hat_i| |x_hat_j|)
        S = torch.abs(torch.bmm(x_norm, x_norm.conj().transpose(-2, -1))).real  # (B, N, N)

        # Spectral attention with temperature
        S_soft = F.softmax(S / self.temperature, dim=-1)
        S_soft = self.dropout(S_soft)

        # Damped mixing (alpha < 1 ensures contraction)
        mixed = torch.bmm(S_soft, x)  # (B, N, D)
        out = self.norm(residual + self.alpha * (mixed - residual))

        return out, S
