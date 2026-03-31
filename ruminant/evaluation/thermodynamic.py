"""
Thermodynamic Evaluator

Measures model quality through information-theoretic metrics derived
from the four-chamber thermodynamics: temperature, entropy, free energy,
and S-entropy coordinates.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ThermodynamicMetrics:
    """Metrics for a single evaluation."""
    information_density: float    # H(output) / H(input)
    phase_coherence: float        # mean spectral correlation
    convergence_rate: float       # contraction constant c
    rumination_depth: int         # cycles to convergence
    chamber_temperatures: dict    # T per chamber
    total_free_energy: float      # F = U - TS
    s_entropy: tuple              # (S_k, S_t, S_e) coordinates


class ThermodynamicEvaluator:
    """Evaluates model quality using thermodynamic metrics.

    Unlike perplexity (which measures token-level prediction quality),
    thermodynamic metrics measure structural properties:
    - Does the model converge? (contraction rate)
    - Does it detect hidden correlations? (spectral coherence)
    - Does it fill knowledge gaps? (gap reduction)
    - Does it reach equilibrium? (free energy minimisation)
    """

    def evaluate_representations(self, X_input: np.ndarray, X_output: np.ndarray,
                                  spectral_matrix: np.ndarray = None) -> ThermodynamicMetrics:
        """Evaluate a forward pass through thermodynamic lens."""

        # Information density: ratio of output to input entropy
        H_in = self._entropy(X_input)
        H_out = self._entropy(X_output)
        info_density = H_out / (H_in + 1e-8)

        # Phase coherence: mean off-diagonal spectral correlation
        if spectral_matrix is not None:
            n = spectral_matrix.shape[0]
            mask = ~np.eye(n, dtype=bool)
            phase_coherence = float(np.mean(np.abs(spectral_matrix[mask])))
        else:
            phase_coherence = 0.0

        # Convergence rate: how much the representation changed
        dist = np.linalg.norm(X_output - X_input, 'fro')
        scale = np.linalg.norm(X_input, 'fro') + 1e-8
        convergence_rate = float(dist / scale)

        # Chamber temperatures (variance at different processing stages)
        T_input = float(np.var(X_input))
        T_output = float(np.var(X_output))

        # Free energy
        U = 0.5 * float(np.sum(X_output ** 2))
        S = H_out
        F = U - T_output * S

        # S-entropy coordinates
        S_k = info_density  # knowledge entropy
        S_t = 1 - abs(np.corrcoef(X_output.flatten()[:100], X_output.flatten()[1:101])[0, 1]) if X_output.size > 101 else 0.5
        S_e = convergence_rate

        return ThermodynamicMetrics(
            information_density=round(info_density, 6),
            phase_coherence=round(phase_coherence, 6),
            convergence_rate=round(convergence_rate, 6),
            rumination_depth=1,
            chamber_temperatures={"input": round(T_input, 6), "output": round(T_output, 6)},
            total_free_energy=round(F, 4),
            s_entropy=(round(S_k, 4), round(S_t, 4), round(S_e, 4)),
        )

    def _entropy(self, X: np.ndarray) -> float:
        """Shannon entropy of the representation energy distribution."""
        energy = np.abs(X.flatten()) ** 2
        total = energy.sum() + 1e-15
        p = energy / total
        return float(-np.sum(p * np.log(p + 1e-15)))

    def compare_models(self, metrics_a: ThermodynamicMetrics,
                       metrics_b: ThermodynamicMetrics) -> dict:
        """Compare two models using thermodynamic criteria."""
        return {
            "info_density_winner": "A" if metrics_a.information_density > metrics_b.information_density else "B",
            "coherence_winner": "A" if metrics_a.phase_coherence > metrics_b.phase_coherence else "B",
            "convergence_winner": "A" if metrics_a.convergence_rate < metrics_b.convergence_rate else "B",
            "free_energy_winner": "A" if metrics_a.total_free_energy < metrics_b.total_free_energy else "B",
        }
