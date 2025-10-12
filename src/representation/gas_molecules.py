"""
Gas Molecular Dynamics Representation
Models transaction networks as gas molecules with harmonic interference.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from scipy.fft import fft, fftfreq


@dataclass
class Molecule:
    """Financial entity as gas molecule."""
    id: str
    position: np.ndarray  # Position in S-entropy space (3D)
    velocity: np.ndarray  # Velocity in S-space
    wavefunction_amplitude: float  # |A_i|
    frequency: float  # ω_i (fundamental transaction frequency)
    phase: float  # φ_i
    wavevector: np.ndarray  # k_i (3D)


class FinancialGasSystem:
    """
    Financial transaction network as gas molecular system.
    Based on: financial-representation.tex Gas Molecular section
    """
    
    def __init__(self, market_temperature: float = 1.0):
        """
        Initialize gas system.
        
        Args:
            market_temperature: T_market for Maxwell-Boltzmann distribution
        """
        self.molecules: Dict[str, Molecule] = {}
        self.market_temperature = market_temperature
        self.k_boltzmann = 1.0  # Normalized
    
    def add_molecule(self, 
                     molecule_id: str,
                     position: np.ndarray,
                     velocity: np.ndarray,
                     amplitude: float,
                     frequency: float,
                     phase: float = 0.0,
                     wavevector: np.ndarray = None):
        """
        Add molecule to system.
        
        Args:
            molecule_id: Unique identifier
            position: (S_knowledge, S_time, S_entropy) coordinates
            velocity: Velocity in S-space
            amplitude: Wavefunction amplitude
            frequency: Fundamental transaction frequency
            phase: Initial phase
            wavevector: 3D wave vector (default: random)
        """
        if wavevector is None:
            wavevector = np.random.randn(3) * 0.1
        
        molecule = Molecule(
            id=molecule_id,
            position=position,
            velocity=velocity,
            wavefunction_amplitude=amplitude,
            frequency=frequency,
            phase=phase,
            wavevector=wavevector
        )
        self.molecules[molecule_id] = molecule
    
    def wavefunction(self, molecule_id: str, r: np.ndarray, t: float) -> complex:
        """
        Calculate wavefunction for molecule at position r and time t.
        
        Ψ_i(r,t) = A_i * exp(i(k_i·r - ω_i*t + φ_i))
        
        Args:
            molecule_id: Molecule identifier
            r: Position vector (3D)
            t: Time
        
        Returns:
            Complex wavefunction value
        """
        mol = self.molecules[molecule_id]
        phase = np.dot(mol.wavevector, r) - mol.frequency * t + mol.phase
        return mol.wavefunction_amplitude * np.exp(1j * phase)
    
    def detect_harmonic_coincidence(self, 
                                    mol_i: str, 
                                    mol_j: str,
                                    epsilon_tol: float = 0.05,
                                    max_n: int = 5) -> Tuple[bool, int, int, float]:
        """
        Detect harmonic coincidence between two molecules.
        
        |n*ω_i - m*ω_j| < ε_tol
        
        Args:
            mol_i, mol_j: Molecule IDs
            epsilon_tol: Tolerance for coincidence
            max_n: Maximum harmonic order to check
        
        Returns:
            (coincidence_found, n, m, correlation_strength)
        """
        omega_i = self.molecules[mol_i].frequency
        omega_j = self.molecules[mol_j].frequency
        
        for n in range(1, max_n + 1):
            for m in range(1, max_n + 1):
                diff = abs(n * omega_i - m * omega_j)
                if diff < epsilon_tol:
                    # Calculate correlation strength
                    correlation = 1.0 - (diff / epsilon_tol)
                    return True, n, m, correlation
        
        return False, 0, 0, 0.0
    
    def calculate_correlation(self, mol_i: str, mol_j: str, 
                             t_samples: np.ndarray,
                             r_point: np.ndarray = None) -> float:
        """
        Calculate correlation ρ_ij = |⟨Ψ_i|Ψ_j⟩|
        
        Args:
            mol_i, mol_j: Molecule IDs
            t_samples: Time samples for integration
            r_point: Spatial point (default: origin)
        
        Returns:
            Correlation coefficient [0, 1]
        """
        if r_point is None:
            r_point = np.zeros(3)
        
        # Calculate inner product ⟨Ψ_i|Ψ_j⟩
        inner_product = 0.0
        for t in t_samples:
            psi_i = self.wavefunction(mol_i, r_point, t)
            psi_j = self.wavefunction(mol_j, r_point, t)
            inner_product += np.conj(psi_i) * psi_j
        
        inner_product /= len(t_samples)
        return abs(inner_product)
    
    def build_correlation_network(self, 
                                  epsilon_tol: float = 0.05,
                                  correlation_threshold: float = 0.7) -> Dict[Tuple[str, str], float]:
        """
        Build correlation network from harmonic coincidences.
        
        Args:
            epsilon_tol: Harmonic coincidence tolerance
            correlation_threshold: Minimum correlation to include edge
        
        Returns:
            Dictionary of (mol_i, mol_j) -> correlation_strength
        """
        edges = {}
        molecule_ids = list(self.molecules.keys())
        
        for i, mol_i in enumerate(molecule_ids):
            for mol_j in molecule_ids[i+1:]:
                found, n, m, strength = self.detect_harmonic_coincidence(
                    mol_i, mol_j, epsilon_tol
                )
                if found and strength >= correlation_threshold:
                    edges[(mol_i, mol_j)] = strength
        
        return edges
    
    def maxwell_boltzmann_distribution(self, values: np.ndarray) -> np.ndarray:
        """
        Calculate Maxwell-Boltzmann probability distribution.
        
        P(V) = (1/Z) * exp(-β*V)
        
        Args:
            values: Array of node values
        
        Returns:
            Probability distribution
        """
        beta = 1.0 / (self.k_boltzmann * self.market_temperature)
        unnormalized = np.exp(-beta * values)
        Z = np.sum(unnormalized)  # Partition function
        return unnormalized / Z if Z > 0 else unnormalized
    
    def calculate_free_energy(self, values: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Calculate Helmholtz free energy F = ⟨E⟩ - T*S
        
        Args:
            values: Energy values
            probabilities: Probability distribution
        
        Returns:
            Free energy
        """
        avg_energy = np.dot(values, probabilities)
        
        # Entropy S = -Σ P*log(P)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        free_energy = avg_energy - self.market_temperature * entropy
        return free_energy
    
    def propagate_reality_wave(self, 
                               r: np.ndarray, 
                               t: float,
                               coefficients: Dict[str, float] = None) -> complex:
        """
        Calculate overall reality wave as superposition.
        
        Ψ_reality(r,t) = Σ c_i * Ψ_i(r,t)
        
        Args:
            r: Position
            t: Time
            coefficients: Importance coefficients for each molecule
        
        Returns:
            Reality wave value
        """
        if coefficients is None:
            # Equal importance
            coefficients = {mol_id: 1.0 for mol_id in self.molecules.keys()}
        
        reality_wave = 0.0 + 0.0j
        for mol_id, coeff in coefficients.items():
            if mol_id in self.molecules:
                reality_wave += coeff * self.wavefunction(mol_id, r, t)
        
        return reality_wave
    
    def update_positions(self, dt: float):
        """
        Update molecule positions based on velocities.
        
        Args:
            dt: Time step
        """
        for molecule in self.molecules.values():
            molecule.position += molecule.velocity * dt
