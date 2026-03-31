"""
Rumination Loop: Iterative Fixed-Point Computation

Composes the four chambers into a single forward pass and iterates
until convergence (completeness > threshold) or max cycles reached.
"""

import torch
import torch.nn as nn
from ruminant.chambers.rumen import RumenLayer
from ruminant.chambers.reticulum import ReticulumLayer
from ruminant.chambers.omasum import OmasumLayer
from ruminant.chambers.abomasum import AbomasumLayer


class RuminationLoop(nn.Module):
    """The four-chamber model with iterative rumination.

    One rumination cycle: Rumen -> Reticulum -> Omasum -> Abomasum.
    Iteration continues until the representation converges to a fixed point
    (guaranteed by the contraction mapping property).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_rumen_layers: int = 2,
        n_reticulum_layers: int = 1,
        n_omasum_layers: int = 1,
        n_abomasum_layers: int = 1,
        max_rumination_cycles: int = 7,
        completeness_threshold: float = 0.95,
        spectral_alpha: float = 0.3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_cycles = max_rumination_cycles
        self.threshold = completeness_threshold

        # Chamber 1: multiple rumen layers for deep mixing
        self.rumen = nn.ModuleList([
            RumenLayer(d_model, n_heads, dropout)
            for _ in range(n_rumen_layers)
        ])

        # Chamber 2: spectral filtering
        self.reticulum = nn.ModuleList([
            ReticulumLayer(d_model, spectral_alpha, dropout=dropout)
            for _ in range(n_reticulum_layers)
        ])

        # Chamber 3: graph completion
        self.omasum = nn.ModuleList([
            OmasumLayer(d_model, dropout=dropout)
            for _ in range(n_omasum_layers)
        ])

        # Chamber 4: refinement
        self.abomasum = nn.ModuleList([
            AbomasumLayer(d_model, n_heads, dropout)
            for _ in range(n_abomasum_layers)
        ])

    def _completeness(self, x: torch.Tensor, x_prev: torch.Tensor) -> float:
        """Measure how close we are to the fixed point."""
        dist = torch.norm(x - x_prev, p='fro')
        scale = torch.norm(x, p='fro') + 1e-8
        return 1.0 - (dist / scale).item()

    def forward_one_cycle(self, x: torch.Tensor) -> dict:
        """One complete pass through all four chambers."""
        # Chamber 1: Rumen
        for layer in self.rumen:
            x = layer(x)

        # Chamber 2: Reticulum
        S = None
        for layer in self.reticulum:
            x, S = layer(x)

        # Chamber 3: Omasum
        unc_before = unc_after = None
        for layer in self.omasum:
            x, unc_before, unc_after = layer(x, S)

        # Chamber 4: Abomasum
        for layer in self.abomasum:
            x = layer(x, unc_after)

        return {
            "output": x,
            "spectral_matrix": S,
            "uncertainty_before": unc_before,
            "uncertainty_after": unc_after,
        }

    def forward(self, x: torch.Tensor) -> dict:
        """Full rumination: iterate until convergence or max cycles."""
        history = []
        x_current = x

        for cycle in range(self.max_cycles):
            x_prev = x_current.detach().clone()
            result = self.forward_one_cycle(x_current)
            x_current = result["output"]

            completeness = self._completeness(x_current, x_prev)
            history.append({
                "cycle": cycle,
                "completeness": completeness,
                "mean_uncertainty": result["uncertainty_after"].mean().item() if result["uncertainty_after"] is not None else 0,
            })

            if completeness > self.threshold:
                break

        return {
            "output": x_current,
            "history": history,
            "n_cycles": len(history),
            "converged": completeness > self.threshold,
            "spectral_matrix": result["spectral_matrix"],
        }
