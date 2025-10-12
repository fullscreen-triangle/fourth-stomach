"""
Shadow Network and Representational Equivalence
Transforms between Circuit, Sequence, and Gas representations.
"""

from typing import Dict, Tuple, List
import numpy as np
from scipy.fft import fft, ifft

from circuit import FinancialCircuit
from gas_molecules import FinancialGasSystem, Molecule
from sequence import Direction, Transaction


class ShadowNetwork:
    """
    Virtual transaction network revealing miracle circuits.
    Based on: financial-representation.tex Shadow Network section
    """
    
    def __init__(self, s_threshold: float = 1.0):
        """
        Initialize shadow network.
        
        Args:
            s_threshold: S-value threshold for miracle detection
        """
        self.s_threshold = s_threshold
        self.miracle_edges: Dict[Tuple[str, str], float] = {}
    
    def calculate_s_value(self, 
                         correlation: float,
                         temporal_cost: float,
                         information_gain: float) -> Tuple[float, float, float]:
        """
        Calculate S-entropy coordinates (S_t, S_i, S_e).
        
        Args:
            correlation: Edge correlation |ρ_ij|
            temporal_cost: Time cost of interaction
            information_gain: Information gained from interaction
        
        Returns:
            (S_temporal, S_information, S_entropy)
        """
        # Normalize and invert where appropriate
        s_temporal = 1.0 / (temporal_cost + 1e-6)
        s_information = information_gain
        s_entropy = correlation
        
        return s_temporal, s_information, s_entropy
    
    def is_miracle_circuit(self, s_values: Tuple[float, float, float]) -> bool:
        """
        Check if circuit element exhibits miraculous behavior.
        
        Miracle if any S_x > 1.0
        """
        s_t, s_i, s_e = s_values
        return any(s > self.s_threshold for s in [s_t, s_i, s_e])
    
    def detect_quantum_coherence(self, 
                                 correlation: float,
                                 current_amplitudes: List[complex]) -> float:
        """
        Detect quantum-coherent superposition in shadow edges.
        
        I_shadow = Σ α_k I_k e^(iφ_k)
        
        Args:
            correlation: Shadow edge correlation
            current_amplitudes: Complex current amplitudes
        
        Returns:
            S-value indicating coherence strength
        """
        # Constructive interference amplitude
        total_amplitude = np.sum(current_amplitudes)
        max_individual = np.max([np.abs(amp) for amp in current_amplitudes])
        
        if max_individual > 0:
            amplification = np.abs(total_amplitude) / max_individual
            
            # S-value based on amplification
            s_value = amplification * correlation
            return s_value
        
        return 0.0
    
    def add_shadow_edge(self, from_node: str, to_node: str, 
                       correlation: float,
                       s_values: Tuple[float, float, float]):
        """
        Add shadow edge if miraculous.
        
        Args:
            from_node, to_node: Node IDs
            correlation: Edge correlation
            s_values: S-entropy coordinates
        """
        if self.is_miracle_circuit(s_values):
            self.miracle_edges[(from_node, to_node)] = correlation


class RepresentationTransformer:
    """
    Transforms between Circuit, Sequence, and Gas representations.
    Based on: financial-representation.tex Representational Equivalence
    """
    
    def __init__(self):
        self.information_loss_tolerance = 0.05  # 5% max loss
    
    def circuit_to_sequence(self, circuit: FinancialCircuit) -> List[List[Direction]]:
        """
        T_C→S: Circuit → Sequence
        
        Encodes current flows and voltage differences as directions.
        """
        sequence = []
        
        for (from_node, to_node), element in circuit.elements.items():
            current = circuit.edge_currents.get((from_node, to_node), 0.0)
            
            v_from = circuit.node_potentials.get(from_node, 0.0)
            v_to = circuit.node_potentials.get(to_node, 0.0)
            voltage_drop = v_from - v_to
            
            # Encode as directions
            dirs = []
            
            # Current magnitude -> North/South
            if abs(current) > 1.0:
                dirs.append(Direction.NORTH)
            else:
                dirs.append(Direction.SOUTH)
            
            # Voltage drop -> East/West
            if voltage_drop > 0:
                dirs.append(Direction.EAST)
            else:
                dirs.append(Direction.WEST)
            
            # Power (I*V) -> Up/Down
            power = current * voltage_drop
            if power > 0:
                dirs.append(Direction.UP)
            else:
                dirs.append(Direction.DOWN)
            
            sequence.append(dirs)
        
        return sequence
    
    def sequence_to_gas(self, sequence: List[List[Direction]]) -> FinancialGasSystem:
        """
        T_S→G: Sequence → Gas
        
        Interprets patterns as wavefunction superpositions.
        """
        gas_system = FinancialGasSystem()
        
        for i, directions in enumerate(sequence):
            # Extract frequency from direction patterns
            direction_values = [d.value for d in directions]
            
            # Encode direction sequence as frequency
            # Simple: count transitions
            frequency = len(set(direction_values)) * 0.5  # Base frequency
            
            # Amplitude from pattern complexity
            amplitude = len(directions) / 6.0  # Normalized by max
            
            # Position based on sequence index
            position = np.array([i * 0.1, 0.0, 0.0])
            velocity = np.array([0.1, 0.0, 0.0])
            
            # Random phase
            phase = np.random.rand() * 2 * np.pi
            
            gas_system.add_molecule(
                molecule_id=f"mol_{i}",
                position=position,
                velocity=velocity,
                amplitude=amplitude,
                frequency=frequency,
                phase=phase
            )
        
        return gas_system
    
    def gas_to_circuit(self, gas_system: FinancialGasSystem, 
                      t_sample: float = 0.0) -> FinancialCircuit:
        """
        T_G→C: Gas → Circuit
        
        Maps molecular interference patterns to circuit currents.
        """
        circuit = FinancialCircuit()
        
        molecule_ids = list(gas_system.molecules.keys())
        
        # Add nodes with potentials from wavefunction amplitudes
        for mol_id, molecule in gas_system.molecules.items():
            potential = molecule.wavefunction_amplitude * np.cos(molecule.phase)
            circuit.add_node(mol_id, net_worth=potential)
        
        # Add edges where harmonic coincidence exists
        correlation_network = gas_system.build_correlation_network(
            epsilon_tol=0.05,
            correlation_threshold=0.7
        )
        
        for (mol_i, mol_j), correlation in correlation_network.items():
            # Calculate current from interference
            r_point = np.zeros(3)
            psi_i = gas_system.wavefunction(mol_i, r_point, t_sample)
            psi_j = gas_system.wavefunction(mol_j, r_point, t_sample)
            
            # Current from interference pattern
            interference = psi_i * np.conj(psi_j)
            current = np.real(interference)
            
            # Resistance inversely proportional to correlation
            resistance = 1.0 / (correlation + 0.1)
            
            circuit.add_resistor(mol_i, mol_j, resistance)
            circuit.set_current(mol_i, mol_j, current)
        
        return circuit
    
    def measure_information_preservation(self, 
                                        original_sequence: List[List[Direction]],
                                        reconstructed_sequence: List[List[Direction]]) -> float:
        """
        Measure information loss through transformation cycle.
        
        I(T) should equal I(C) = I(S) = I(G)
        
        Returns:
            Information preservation ratio [0, 1]
        """
        # Convert to comparable format
        original_flat = [d.value for seq in original_sequence for d in seq]
        reconstructed_flat = [d.value for seq in reconstructed_sequence for d in seq]
        
        # Pad to same length
        max_len = max(len(original_flat), len(reconstructed_flat))
        original_flat += ['_'] * (max_len - len(original_flat))
        reconstructed_flat += ['_'] * (max_len - len(reconstructed_flat))
        
        # Count matches
        matches = sum(1 for o, r in zip(original_flat, reconstructed_flat) if o == r)
        preservation = matches / max_len if max_len > 0 else 1.0
        
        return preservation
    
    def full_cycle_transform(self, transactions: List[Transaction]) -> Dict[str, any]:
        """
        Test full transformation cycle: T → C → S → G → C' → T'
        
        Args:
            transactions: Original transactions
        
        Returns:
            Dictionary with transformation results and information loss
        """
        from sequence import SequenceEncoder
        
        # T → S: Transaction to Sequence
        encoder = SequenceEncoder()
        original_sequence = encoder.encode_stream(transactions)
        
        # S → G: Sequence to Gas
        gas_system = self.sequence_to_gas(original_sequence)
        
        # G → C: Gas to Circuit
        circuit = self.gas_to_circuit(gas_system)
        
        # C → S: Circuit back to Sequence
        reconstructed_sequence = self.circuit_to_sequence(circuit)
        
        # Measure preservation
        preservation = self.measure_information_preservation(
            original_sequence,
            reconstructed_sequence
        )
        
        information_loss = 1.0 - preservation
        
        return {
            'original_sequence': original_sequence,
            'reconstructed_sequence': reconstructed_sequence,
            'gas_system': gas_system,
            'circuit': circuit,
            'information_preservation': preservation,
            'information_loss': information_loss,
            'passes_threshold': information_loss < self.information_loss_tolerance
        }
